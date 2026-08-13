#!/usr/bin/env python3
"""
Инференс GreenVLA для Unitree G1.
Полный state (48) в правильном порядке:
arm_state (14) + leg_state (15) + hand_state (14) + imu_rpy (3) + odometry_xy (2)
Изображение подготавливается вручную с нормализацией ImageNet.
Управление – 28 суставов (руки+кисти) в порядке, соответствующем модели.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
import mujoco
import imageio
from rich.console import Console
from rich.table import Table
from lerobot.common.policies.factory import load_pretrained_policy
from lerobot.common.utils.torch_observation import (
    move_dict_to_batch_for_inference,
    torch_preprocess_dict_inference,
)

console = Console()

# Порядок для логирования (28 суставов, совпадает с joint_map_names)
JOINT_NAMES = [
    "L_Sh_Pitch", "L_Sh_Roll", "L_Sh_Yaw", "L_Elbow",
    "L_Wr_Roll", "L_Wr_Pitch", "L_Wr_Yaw",
    "L_Thumb_0", "L_Thumb_1", "L_Thumb_2",
    "L_Middle_0", "L_Middle_1", "L_Index_0", "L_Index_1",
    "R_Sh_Pitch", "R_Sh_Roll", "R_Sh_Yaw", "R_Elbow",
    "R_Wr_Roll", "R_Wr_Pitch", "R_Wr_Yaw",
    "R_Thumb_0", "R_Thumb_1", "R_Thumb_2",
    "R_Index_0", "R_Index_1", "R_Middle_0", "R_Middle_1"
]

def quat_to_euler(w, x, y, z):
    """Преобразует кватернион (w,x,y,z) в углы Эйлера (roll, pitch, yaw)."""
    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)
    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

class G1MuJoCoEnv:
    def __init__(self, xml_path: str, render_width=640, render_height=480):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, render_height, render_width)

        self.cam_id = 0
        for i in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name == "head_camera":
                self.cam_id = i
                console.print(f"[green]Найдена камера 'head_camera' (ID: {i})[/green]")
                break

        # --- Сопоставление для управления (актуаторы) ---
        # Порядок: левая рука → левая кисть → правая рука → правая кисть (как в исходном скрипте)
        self.actuator_mapping = {}
        joint_act_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_index_0_joint", "left_hand_index_1_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint"
        ]
        count = 0
        for name in joint_act_names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id != -1:
                self.actuator_mapping[count] = act_id
                count += 1
        if count == 28:
            console.print("[green]Успешно сопоставлены 28 актуаторов (руки+кисти).[/green]")
        else:
            console.print(f"[yellow]Сопоставлено {count} из 28.[/yellow]")

        # --- Сопоставление для чтения qpos всех суставов ---
        # arm_state (14)
        self.arm_qpos_indices = []
        arm_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]
        for name in arm_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id != -1:
                self.arm_qpos_indices.append(self.model.jnt_qposadr[joint_id])

        # leg_state (15)
        self.leg_qpos_indices = []
        leg_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"
        ]
        for name in leg_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id != -1:
                self.leg_qpos_indices.append(self.model.jnt_qposadr[joint_id])

        # hand_state (14)
        self.hand_qpos_indices = []
        hand_names = [
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint", "left_hand_middle_1_joint",
            "left_hand_index_0_joint", "left_hand_index_1_joint",
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_index_1_joint",
            "right_hand_middle_0_joint", "right_hand_middle_1_joint"
        ]
        for name in hand_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id != -1:
                self.hand_qpos_indices.append(self.model.jnt_qposadr[joint_id])

        # Проверка
        total = len(self.arm_qpos_indices) + len(self.leg_qpos_indices) + len(self.hand_qpos_indices)
        if total == 43:
            console.print("[green]Собраны индексы всех 43 суставов (arm+leg+hand).[/green]")
        else:
            console.print(f"[yellow]Собрано {total} из 43 суставов.[/yellow]")

        # Для вычисления imu_rpy и odometry_xy будем использовать положение таза
        # pelvis – это корневое тело с именем "pelvis"; его qpos занимает первые 7 слотов (x,y,z,w,x,y,z)
        # В модели они заданы через freejoint (floating_base_joint)
        # В вашем XML freejoint закомментирован, поэтому pelvis может быть зафиксирован.
        # Если freejoint закомментирован, то qpos[0:7] будут нулевыми или фиксированными.
        # В любом случае мы их читаем.

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        qpos = self.data.qpos

        # Суставы
        arm_state = np.array([qpos[idx] for idx in self.arm_qpos_indices], dtype=np.float32)
        leg_state = np.array([qpos[idx] for idx in self.leg_qpos_indices], dtype=np.float32)
        hand_state = np.array([qpos[idx] for idx in self.hand_qpos_indices], dtype=np.float32)

        # Положение и ориентация таза (pelvis)
        pelvis_pos = qpos[0:3]   # x, y, z
        pelvis_quat = qpos[3:7]  # w, x, y, z

        # IMU rpy из кватерниона
        roll, pitch, yaw = quat_to_euler(*pelvis_quat)

        # Сборка полного state_48
        state_48 = np.zeros(48, dtype=np.float32)
        state_48[0:14] = arm_state
        state_48[14:29] = leg_state
        state_48[29:43] = hand_state
        state_48[43] = roll
        state_48[44] = pitch
        state_48[45] = yaw
        state_48[46] = pelvis_pos[0]  # x
        state_48[47] = pelvis_pos[1]  # y

        # Изображение
        self.renderer.update_scene(self.data, camera=self.cam_id)
        image = self.renderer.render()
        return {"state": state_48, "image": image}

    def step(self, action):
        # action – 28 значений для рук+кистей
        for idx in range(28):
            if idx in self.actuator_mapping:
                self.data.ctrl[self.actuator_mapping[idx]] = float(action[idx])

        # 10 шагов физики для стабильности
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        return self._get_obs()


def run_inference(model_path, xml_path, num_steps=500, video_path="g1_output.mp4", prompt="stack the cubes"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[cyan]Загрузка модели: {model_path}[/cyan]")

    policy, input_transforms, output_transforms = load_pretrained_policy(
        model_path,
        data_config_name="g1_r1_loco_2b",
        config_overrides={"device": device}
    )
    policy.to(device).eval()

    console.print(f"[cyan]Инициализация среды...[/cyan]")
    env = G1MuJoCoEnv(xml_path)
    obs = env.reset()

    console.print(f"[bold]Промпт:[/bold] {prompt}\n")
    video_writer = imageio.get_writer(video_path, fps=30)

    prev_action = None
    alpha = 1.85  # Коэффициент сглаживания (можно изменить или отключить)

    # Параметры нормализации изображения (ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    console.print("[yellow]Запуск цикла инференса...[/yellow]\n")

    for step in range(num_steps):
        start_time = time.time()

        # obs["state"] уже 48 – используем напрямую
        raw_obs = {
            "prompt": prompt,
            "observation.state": obs["state"],           # правильный 48-мерный state
            "observation.image.egocentric": obs["image"],
        }

        # Входные трансформы
        transformed_obs = input_transforms(raw_obs)
        preprocessed = torch_preprocess_dict_inference(transformed_obs)
        batch = move_dict_to_batch_for_inference(preprocessed, device=device)

        # Ручная подготовка изображения (гарантированно)
        img_np = obs["image"]
        img_tensor = torch.from_numpy(img_np).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        img_tensor = F.interpolate(img_tensor, size=(448, 448), mode="bilinear", align_corners=False)
        img_tensor = (img_tensor - mean) / std
        batch["image"] = {"egocentric": img_tensor}

        # Инференс
        with torch.inference_mode():
            raw_actions = policy.select_action(batch)

        first_step_action = raw_actions[:, 0:1, :]
        actions_dict = output_transforms({
            "actions": first_step_action.cpu().numpy(),
            "state": batch["state"].cpu().numpy()
        })
        action = actions_dict["actions"][0, 0, :28]

        # Сглаживание (если нужно)
        if prev_action is not None:
            action = alpha * action + (1 - alpha) * prev_action
        prev_action = action.copy()

        # Шаг симуляции
        obs = env.step(action)
        video_writer.append_data(obs["image"])

        # Логирование
        if step % 50 == 0:
            elapsed = time.time() - start_time
            table = Table(title=f"Шаг {step}/{num_steps} (Time: {elapsed:.3f}s)")
            table.add_column("Сустав", style="cyan")
            table.add_column("Action (rad)", justify="right", style="green")
            for name, val in zip(JOINT_NAMES, action):
                table.add_row(name, f"{val:.3f}")
            console.print(table)

    video_writer.close()
    console.print(f"\n[bold green]Готово: {video_path}[/bold green]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/human/Kruglov/GreenVLA/outputs/g1_r1_2b_v3/checkpoints/last")
    parser.add_argument("--xml-path", type=str, default="/home/human/Kruglov/g1_model/unitree_g1/scene_table_object.xml")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--video-path", type=str, default="g1_output.mp4")
    parser.add_argument("--prompt", type=str, default="stack the cubes")
    args = parser.parse_args()

    run_inference(args.model_path, args.xml_path, args.num_steps, args.video_path, args.prompt)
