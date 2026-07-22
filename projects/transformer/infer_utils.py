"""Inference utilities for the Transformer tracker."""

from __future__ import annotations

import numpy as np

from tracking.infer_utils import (  # noqa: F401
    EMA_ALPHA,
    G1TrackInferFn,
    G1TrackMjSim,
    NUM_ACTION,
    NUM_STATE,
    State,
    apply_ema_qpos,
    g1_infer_env_config,
)


class G1TrackTransformerInferFn(G1TrackInferFn):
    """G1TrackInferFn + rolling K-frame obs buffer for the Transformer policy."""

    def __init__(self, *args, history_len: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        if history_len < 1:
            raise ValueError(f"history_len must be >= 1, got {history_len}")
        self.history_len = int(history_len)
        self._history_buffer: np.ndarray | None = None

    def reset_history(self) -> None:
        """Drop the buffer; the next obs re-seeds it, matching ObsHistoryWrapper.reset."""
        self._history_buffer = None
        self.info["last_action"][:] = 0.0
        self.info["nn_action"][:] = 0.0
        self.info["motor_targets"][:] = self._nom_jnt_qpos

    def _push_history(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        assert obs.shape[0] == self.num_envs, (
            f"history push got num_envs={obs.shape[0]}, expected {self.num_envs}"
        )
        if self._history_buffer is None:
            self._history_buffer = np.broadcast_to(
                obs[:, None, :],
                (self.num_envs, self.history_len, obs.shape[-1]),
            ).copy()
        else:
            if self.history_len > 1:
                self._history_buffer[:, :-1] = self._history_buffer[:, 1:]
            self._history_buffer[:, -1] = obs
        return self._history_buffer.copy()

    def infer_onnx(self, state: State, ref_state) -> np.ndarray:
        ref_next = ref_state.get("ref_next", ref_state["ref_curr"])
        self.update_state(state, ref_state)
        if self.privileged:
            obs = self.get_nn_priv_state(self.info, ref_next, self.info["last_action"])
        else:
            obs = self.get_nn_state(self.info, ref_next, self.info["last_action"])

        stacked = self._push_history(obs)
        nn_action = self.nn_policy.infer(stacked)

        motor_targets = self.nn2motor_action(nn_action)
        self.info["motor_targets"] = motor_targets.copy()
        self.info["step"] += 1
        self.info["nn_action"] = nn_action
        self.info["last_action"] = nn_action.copy()
        return motor_targets

    def infer_onnx_real(
        self,
        root_quat: np.ndarray,
        root_gyro: np.ndarray,
        jnt_qpos: np.ndarray,
        jnt_qvel: np.ndarray,
        ref_state: dict,
    ) -> np.ndarray:
        """Real-robot inference (skips mj_forward on the robot state)."""
        from utils.transforms_np import quat2mat

        ref_next = ref_state.get("ref_next", ref_state["ref_curr"])

        qpos = np.zeros((1, self.mj_model.nq), dtype=np.float32)
        qpos[0, :3] = [0.0, 0.0, 0.78]
        qpos[0, 3:7] = root_quat
        qpos[0, 7:] = jnt_qpos

        qvel = np.zeros((1, self.mj_model.nv), dtype=np.float32)
        qvel[0, 3:6] = root_gyro
        qvel[0, 6:] = jnt_qvel

        pelvis2world_rot = quat2mat(root_quat[None])
        gvec_pelvis = -pelvis2world_rot.transpose(0, 2, 1)[..., 2]

        self.info["gyro_pelvis"][:] = root_gyro[None]
        self.info["gvec_pelvis"][:] = gvec_pelvis
        self.info["qpos"][:] = qpos
        self.info["qvel"][:] = qvel
        self.update_coord_cmd(ref_state)

        obs = self.get_nn_state(self.info, ref_next, self.info["last_action"])
        stacked = self._push_history(obs)
        nn_action = self.nn_policy.infer(stacked)

        motor_targets = self.nn2motor_action(nn_action)
        self.info["motor_targets"] = motor_targets.copy()
        self.info["step"] += 1
        self.info["nn_action"] = nn_action
        self.info["last_action"] = nn_action.copy()
        return motor_targets
