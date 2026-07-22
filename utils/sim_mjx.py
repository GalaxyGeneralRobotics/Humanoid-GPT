import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import mujoco.viewer
import numpy as np

from utils.hand import load_mj_model
from utils.sim_base import BaseSim, State


def get_sensor_data(
    model: mujoco.MjModel, data: mjx.Data, sensor_name: str
) -> jax.Array:
    """Gets sensor data given sensor name."""
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[:, sensor_adr : sensor_adr + sensor_dim]


def get_collision_distance(contact, geom1: int, geom2: int) -> np.ndarray:
    geom = contact.geom  # (..., N, 2)
    dist = contact.dist  # (..., N)
    mask = (geom == np.array([geom1, geom2])).all(-1) | (
        geom == np.array([geom2, geom1])
    ).all(-1)  # (..., N)
    masked_dist = np.where(mask, dist, np.inf)  # (..., N)
    nearest = masked_dist.min(axis=-1)  # (...,)
    return np.where(np.isfinite(nearest), nearest, 0.0)


def geoms_colliding(mj_data, geom1: int, geom2: int):
    """
    Vectorised collision predicate.

    Returns
    -------
    np.ndarray[bool] with shape equal to mj_data.contact.dist.shape[:-1]
    """
    if mj_data.contact.geom.size == 0:  # handles empty batches
        batch_shape = mj_data.contact.dist.shape[:-1]
        return np.zeros(batch_shape, dtype=bool)

    dist = get_collision_distance(mj_data.contact, geom1, geom2)
    return dist < 0


def _make_torque_step(model, n_sub):
    @jax.jit
    def _step(
        data: mjx.Data,
        kps: jax.Array,
        kds: jax.Array,
        torque_lim: jax.Array,
        qpos_des: jax.Array,
    ):
        def body(d, _):
            pos_err = qpos_des - d.qpos[7:]
            vel_err = -d.qvel[6:]
            torque = kps * pos_err + kds * vel_err
            torque = jnp.clip(torque, -torque_lim, torque_lim)
            d = mjx.step(model, d.replace(ctrl=torque))
            return d, None

        data, _ = jax.lax.scan(body, data, None, length=n_sub)
        return data

    return _step


def _make_reset_fn(model, nv):
    """JIT-compiled reset for one robot; returns mjx.Data on device."""

    @jax.jit
    def _reset(_, qpos0):
        d = mjx.make_data(model).replace(qpos=qpos0, qvel=jnp.zeros(nv, jnp.float32))
        return mjx.forward(model, d)

    return _reset


class MJXSim(BaseSim):
    kps: jax.Array
    kds: jax.Array
    torque_limit: jax.Array
    init_qpos: jax.Array

    def __init__(
        self,
        xml_path: str,
        num_envs: int,
        ctrl_dt: float = 0.02,
        sim_dt: float = 0.001,
        episode_length=1000,
        headless: bool = False,
    ):
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.ctrl_dt = ctrl_dt
        self.sim_dt = sim_dt
        self.headless = headless
        self.n_sub = int(ctrl_dt / sim_dt)

        # load MuJoCo once (hand-aware: mounts the HAND end-effector, rigid)
        self.mj_model = load_mj_model(xml_path)
        self.mj_model.opt.timestep = sim_dt
        self.mjx_model = mjx.put_model(self.mj_model)

        # compile per-robot kernels **once**
        torque_step_1env = _make_torque_step(self.mjx_model, self.n_sub)
        reset_1env = _make_reset_fn(self.mjx_model, self.mjx_model.nv)

        # vectorise --> [num_envs, …]  and jit again
        self._step_fn = jax.jit(jax.vmap(torque_step_1env, in_axes=(0, 0, 0, 0, 0)))
        self._reset_fn = jax.jit(jax.vmap(reset_1env, in_axes=(0, 0)))

    # ----------  public API  ------------------------------------------------------
    def init_state(self) -> State:
        mjx_data = self._reset_fn(jnp.arange(self.num_envs), self.init_qpos)
        if not self.headless:
            mj_data = mujoco.MjData(self.mj_model)
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, mj_data)
        else:
            mj_data = None

        return State(mj_data=mj_data, mjx_data=mjx_data)

    def reset(self, state: State, sim_ids: np.ndarray | None = None) -> State:
        if sim_ids is None:
            sim_ids = np.arange(self.num_envs)
        mjx_data = state.mjx_data
        sim_ids = jnp.asarray(sim_ids)
        qpos = mjx_data.qpos.at[sim_ids].set(self.init_qpos[sim_ids])
        qvel = mjx_data.qvel.at[sim_ids].set(0.0)
        ctrl = mjx_data.ctrl.at[sim_ids].set(0.0)
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        return State(mj_data=state.mj_data, mjx_data=mjx_data)

    def step(self, state: State, action: np.ndarray) -> State:
        mjx_data = state.mjx_data
        mjx_data = self._step_fn(
            mjx_data,
            self.kps,
            self.kds,
            self.torque_limit,
            jnp.asarray(action, dtype=jnp.float32),
        )
        return State(mj_data=state.mj_data, mjx_data=mjx_data)

    def view(self, state: State, sim_id: int = 0) -> None:
        mj_data = state.mj_data
        mjx_data = state.mjx_data

        if not self.headless:  # cheap host sync for a single env
            mujoco.mjx.get_data_into(
                mj_data,
                self.mj_model,
                jax.tree_map(lambda x: x[sim_id], mjx_data),
            )
            mujoco.mj_forward(self.mj_model, mj_data)
            self.viewer.sync()
        else:
            raise RuntimeError("Viewer is not available in headless mode.")


def torque_step_dr(
    rng: jax.Array,
    model: mjx.Model,
    data: mjx.Data,
    qpos_des: jax.Array,
    kps: jax.Array,
    kds: jax.Array,
    kp_scale: jax.Array,
    kd_scale: jax.Array,
    rfi_lim_scale: jax.Array,
    torque_limit: jax.Array,
    n_substeps: int = 1,
) -> tuple[jax.Array, mjx.Data]:
    def single_step(carry, _):
        rng, data = carry
        rng, rng_rfi = jax.random.split(rng, 2)

        # pd control
        pos_err = qpos_des - data.qpos[7:]
        vel_err = -data.qvel[6:]
        torque = (kp_scale * kps) * pos_err + (kd_scale * kds) * vel_err

        # rfi noise
        rfi_noise = jax.random.uniform(
            rng_rfi, shape=torque.shape, minval=-1.0, maxval=1.0
        )
        torque += rfi_lim_scale * torque_limit * rfi_noise

        # clip
        torque = jnp.clip(torque, -torque_limit, torque_limit)

        # apply torque
        data = data.replace(ctrl=torque)
        data = mjx.step(model, data)

        return (rng, data), None

    return jax.lax.scan(single_step, (rng, data), (), n_substeps)[0]