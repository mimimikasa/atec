import os
import torch
from typing import Any


class AlgSolution:

    ACTION_SCALE = 0.5
    EE_BODY_NAME_CANDIDATES = ("gripper_base", "piper_gripper_base")
    ARM_JOINT_NAME_CANDIDATES = (
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"],
    )

    # ---- Task D dead-reckoning schedule ----
    # Each predict() call = 1 step = decimation(4) × sim_dt(0.005) = 0.02 s
    # lateral > 0 = LEFT, lateral < 0 = RIGHT
    STEP_DT = 0.02

    # Phase 1: back up to get behind the box
    P1_BACKUP_STEPS = 150        # ~3 s
    P1_BACKUP_FWD = -0.5
    P1_BACKUP_LAT = 0.0

    # Phase 2: move LEFT to get to the LEFT side of the box
    P2_LEFT_STEPS = 320          # ~3 s
    P2_LEFT_FWD = 0.0
    P2_LEFT_LAT = 0.5            # positive = left

    # Phase 3: walk FORWARD to align x with the box
    P3_ALIGN_X_STEPS = 80       # ~3 s
    P3_ALIGN_X_FWD = 0.5
    P3_ALIGN_X_LAT = 0.0


    # Phase 4: push box to the RIGHT past the front wall (arm extended)
    P4_PUSH_RIGHT_STEPS = 245    # ~5 s
    P4_PUSH_RIGHT_FWD = 0.0
    P4_PUSH_RIGHT_LAT = -0.5     # negative = right

    # Phase 5: back up behind the box again
    P5_BACKUP_STEPS = 150        # ~3 s
    P5_BACKUP_FWD = -0.5
    P5_BACKUP_LAT = 0.0

    # Phase 6: move RIGHT to align behind the box (same y)
    P6_ALIGN_Y_STEPS = 70       # ~3 s
    P6_ALIGN_Y_FWD = 0.0
    P6_ALIGN_Y_LAT = -0.5        # negative = right

    # Phase 7: push box FORWARD into the pit (arm extended)
    P7_PUSH_FWD_STEPS = 250      # ~5 s (2x speed → half steps)
    P7_PUSH_FWD_FWD = 0.4        # bigger strides to step across the gap
    P7_PUSH_FWD_LAT = 0.0

    # Phase 8: cross the pit
    P8_CROSS_FWD = 0.8           # bigger strides
    P8_CROSS_LAT = 0.0
    # --------------------------------------------

    # Arm push pose: open-loop joint targets for extending the arm forward/down
    # arm_joint1–arm_joint8 (arm_joint7,8 = gripper)
    ARM_PUSH_POSE = [0.0, -1.2, 0.5, -1.5, 0.8, 0.0, 0.0, 0.0]

    def __init__(self):
        policy_path = os.path.dirname(os.path.abspath(__file__)) + '/policy.pt'
        self.device = 'cuda'

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.leg_action_dim = 12
        self.arm_action_dim = 8

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        self.train_to_env_action_scale = torch.tensor(
            [
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        self.env_to_train_action_scale = torch.tensor(
            [
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        # Velocity commands
        self.forward_speed = 0.5
        self.lateral_speed = 0.0
        self.fixed_velocity_commands = torch.tensor(
            [0.5, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)

        # Heading correction state
        self.heading = 0.0
        self.control_dt = self.STEP_DT
        self.heading_kp = 2.0
        self.heading_kd = 0.3

        # Arm actions
        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim),
            device=self.device,
            dtype=torch.float32,
        )

        self.arm_push_action = torch.tensor(
            [self.ARM_PUSH_POSE],
            device=self.device,
            dtype=torch.float32,
        )

        # Step counter for dead-reckoning schedule
        self._step = 0

    def reset(self, **kwargs):
        self.heading = 0.0
        self._step = 0

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {}

    # ------------------------------------------------------------------
    # Unused helpers (kept from original for potential IK use elsewhere)
    # ------------------------------------------------------------------
    def _resolve_joint_ids(self, candidates: tuple[list[str], ...]) -> list[int]:
        last_error = None
        for names in candidates:
            try:
                ids, found_names = self.robot.find_joints(names)
            except ValueError as err:
                last_error = err
                continue
            if len(ids) == len(names):
                if candidates is self.ARM_JOINT_NAME_CANDIDATES:
                    self.arm_joint_names = list(found_names)
                return list(ids)
        raise ValueError(
            f"Cannot resolve required joints from candidates: {candidates}. Last error: {last_error}"
        )

    def _resolve_ee_body_name(self) -> str:
        last_error = None
        for name in self.EE_BODY_NAME_CANDIDATES:
            try:
                body_ids, _ = self.robot.find_bodies(name)
            except ValueError as err:
                last_error = err
                continue
            if len(body_ids) == 1:
                return name
        raise ValueError(
            f"Cannot resolve EE body from candidates: {self.EE_BODY_NAME_CANDIDATES}. Last error: {last_error}"
        )

    def _ensure_cartesian_targets(self):
        self.cartesian_ctrl.reset()

    def _compute_arm_overlay_action(self) -> torch.Tensor:
        self._ensure_cartesian_targets()
        arm_jpos_des = self.cartesian_ctrl.compute_base(
            self.ee_pos_target_b,
            self.ee_quat_target_b,
        )
        full_target = self.robot.data.joint_pos.clone()
        full_target[:, self.arm_ids] = arm_jpos_des
        full_target[:, self.gripper_ids] = self.gripper_open_pos.repeat(full_target.shape[0], 1)
        return (full_target - self.default_joint_pos) / self.ACTION_SCALE

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        """PD heading-hold with configurable forward + lateral speed."""
        num_envs = proprio.shape[0]
        wz = proprio[:, 5].mean().item()

        self.heading += wz * self.control_dt
        self.heading = max(-0.5, min(0.5, self.heading))

        yaw_cmd = -self.heading_kp * self.heading - self.heading_kd * wz
        yaw_cmd = max(-1.0, min(1.0, yaw_cmd))

        cmd = torch.tensor(
            [[self.forward_speed, self.lateral_speed, yaw_cmd]],
            device=self.device,
            dtype=proprio.dtype,
        )
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        return cmd

    def _extract_policy_obs(self, obs, action_dim) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)

        idx = 0
        _base_lin_vel = proprio[:, idx:idx + 3]
        idx += 3
        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3
        _velocity_commands_env = proprio[:, idx:idx + 3]
        idx += 3
        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3
        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]

        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)
        velocity_commands = self._get_velocity_commands(proprio)

        policy_obs = torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )
        return policy_obs

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        """Map 12D leg action → 20D full-body action (legs + zero arm)."""
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(
                f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}"
            )
        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)
        return action_env

    def _apply_schedule(self):
        """Dead-reckoning schedule: advance step counter and set velocity + arm."""
        s = self._step

        p1 = self.P1_BACKUP_STEPS
        p2 = p1 + self.P2_LEFT_STEPS
        p3 = p2 + self.P3_ALIGN_X_STEPS
        p4 = p3 + self.P4_PUSH_RIGHT_STEPS
        p5 = p4 + self.P5_BACKUP_STEPS
        p6 = p5 + self.P6_ALIGN_Y_STEPS
        p7 = p6 + self.P7_PUSH_FWD_STEPS

        if s < p1:
            fwd, lat = self.P1_BACKUP_FWD, self.P1_BACKUP_LAT
            self._arm_target = self.arm_default_action
            self.heading_kp = 2.0
            phase = "p1_backup"
        elif s < p2:
            fwd, lat = self.P2_LEFT_FWD, self.P2_LEFT_LAT
            self._arm_target = self.arm_default_action
            self.heading_kp = 2.0
            phase = "p2_left"
        elif s < p3:
            fwd, lat = self.P3_ALIGN_X_FWD, self.P3_ALIGN_X_LAT
            self._arm_target = self.arm_default_action
            self.heading_kp = 2.0
            phase = "p3_align_x"
        elif s < p4:
            fwd, lat = self.P4_PUSH_RIGHT_FWD, self.P4_PUSH_RIGHT_LAT
            self._arm_target = self.arm_push_action
            self.heading_kp = 5.0
            phase = "p4_push_right"
        elif s < p5:
            fwd, lat = self.P5_BACKUP_FWD, self.P5_BACKUP_LAT
            self._arm_target = self.arm_default_action
            self.heading_kp = 2.0
            phase = "p5_backup"
        elif s < p6:
            fwd, lat = self.P6_ALIGN_Y_FWD, self.P6_ALIGN_Y_LAT
            self._arm_target = self.arm_default_action
            self.heading_kp = 2.0
            phase = "p6_align_y"
        elif s < p7:
            fwd, lat = self.P7_PUSH_FWD_FWD, self.P7_PUSH_FWD_LAT
            self._arm_target = self.arm_push_action
            self.heading_kp = 5.0
            phase = "p7_push_fwd"
        else:
            fwd, lat = self.P8_CROSS_FWD, self.P8_CROSS_LAT
            self._arm_target = self.arm_default_action
            self.heading_kp = 2.0
            phase = "p8_cross"

        self.forward_speed = fwd
        self.lateral_speed = lat

        if s % 50 == 0:
            print(f"[TaskD] step={s} ({s*self.STEP_DT:.1f}s) {phase} "
                  f"fwd={fwd} lat={lat} arm={'push' if self._arm_target is self.arm_push_action else 'zero'}")

        self._step += 1

    def predicts(self, obs, current_score):
        """Dead-reckoning sequence for Task D."""
        # Give up when task complete
        if current_score >= 36:
            return {'action': [], 'giveup': True}

        # Apply dead-reckoning schedule
        self._apply_schedule()

        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        policy_obs = self._extract_policy_obs(obs, action_dim)

        with torch.inference_mode():
            action_train = self.policy(policy_obs)

        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)

        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train, action_dim)

        # Apply scheduled arm action
        action_env[:, self.arm_joint_indices] = self._arm_target.repeat(action_env.shape[0], 1)

        action_env = action_env.cpu().numpy().tolist()
        return {'action': action_env, 'giveup': False}
