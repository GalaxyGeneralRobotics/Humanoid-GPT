from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from utils.hand import load_mj_model
from utils.sim_base import BaseSim, State


def get_qpos_ids(mj_model, names):
    return np.hstack([mj_model.joint(n).qposadr for n in names])


def get_dof_ids(mj_model, names):
    return np.hstack([mj_model.joint(n).dofadr for n in names])


def get_sensor_data(mj_model, mj_data, sensor_name: str) -> np.ndarray:
    """Gets sensor data given sensor name."""
    sensor_id = mj_model.sensor(sensor_name).id
    sensor_adr = mj_model.sensor_adr[sensor_id]
    sensor_dim = mj_model.sensor_dim[sensor_id]
    return mj_data.sensordata[sensor_adr : sensor_adr + sensor_dim]


def get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> tuple[np.ndarray, np.ndarray]:
    """Get the distance and normal of the collision between two geoms."""
    mask = (np.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (np.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = np.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    normal = (dist < 0) * contact.frame[idx, :3]
    return dist, normal


def geoms_colliding(state: mujoco.MjData, geom1: int, geom2: int):
    """Return True if the two geoms are colliding."""
    if len(state.contact) == 0:
        return 0
    return get_collision_info(state.contact, geom1, geom2)[0] < 0


class MJSim(BaseSim):
    kps: np.ndarray
    kds: np.ndarray
    torque_limit: np.ndarray
    init_qpos: np.ndarray

    def __init__(self, xml_path: str, ctrl_dt=0.02, sim_dt=0.001, headless=False):
        self.ctrl_dt = ctrl_dt
        self.sim_dt = sim_dt
        self.num_sim_substeps = int(self.ctrl_dt / self.sim_dt)
        # Hand-aware load: mounts the HAND end-effector (rigid, nq stays 36).
        self.mj_model = load_mj_model(xml_path)
        self.mj_model.opt.timestep = sim_dt
        self.headless = headless

    def init_state(self) -> State:
        mj_data = mujoco.MjData(self.mj_model)

        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, mj_data)
        return State(mj_data=mj_data)

    def reset(self, state: State, sim_ids=None) -> State:
        mj_data = state.mj_data

        mj_data.qpos[:] = self.init_qpos
        mj_data.qvel[:] = 0.0
        mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.mj_model, mj_data)
        return State(mj_data=mj_data)

    def step(self, state: State, action: np.ndarray) -> State:
        mj_data = state.mj_data

        for _ in range(self.num_sim_substeps):
            torques = self.kps * (action - mj_data.qpos[7:]) + self.kds * (
                -mj_data.qvel[6:]
            )
            mj_data.ctrl[:] = np.clip(torques, -self.torque_limit, self.torque_limit)
            mujoco.mj_step(self.mj_model, mj_data)

        return State(mj_data=mj_data)

    def view(self, state: State, sim_id: int = 0) -> None:
        if not self.headless:
            self.viewer.sync()
        else:
            print(
                "Headless mode enabled: no graphical viewer is available for rendering."
            )
