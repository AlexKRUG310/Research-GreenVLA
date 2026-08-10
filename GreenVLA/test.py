#!/usr/bin/env python3
"""
Инференс GreenVLA для Unitree G1 с максимальной плавностью (30 Гц).
Задача: stack the cubes
"""
import time
import numpy as np
import torch
import mujoco
import imageio
from rich.console import Console
from rich.table import Table
from lerobot.common.policies.factory import load_pretrained_policy
from lerobot.common.utils.torch_observation import (
    move_dict_to_batch_for_inference,
    torch_preprocess_dict_inference,
)
import torch.nn.functional as F

console = Console()

JOINT_NAMES = [
    "L_Thumb_0", "L_Thumb_1", "L_Thumb_2", "L_Middle_0", "L_Middle_1", "L_Index_0", "L_Index_1",
    "R_Thumb_0", "R_Thumb_1", "R_Thumb_2", "R_Index_0", "R_Index_1", "R_Middle_0", "R_Middle_1",
    "L_Sh_Pitch", "L_Sh_Roll", "L_Sh_Yaw", "L_Elbow", "L_Wr_Roll", "L_Wr_Pitch", "L_Wr_Yaw",
    "R_Sh_Pitch", "R_Sh_Roll", "R_Sh_Yaw", "R_Elbow", "R_Wr_Roll", "R_Wr_Pitch", "R_Wr_Yaw"
]

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
        
        # Автоматическое сопоставление суставов по именам
        self.actuator_mapping = {}
        joint_map_names = [
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
        for name in joint_map_names:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id != -1:
                self.actuator_mapping[count] = act_id
                count += 1
        
        if count == 28:
            console.print(f"[green]Успешно сопоставлено 28 суставов рук.[/green]")
        else:
            console.print(f"[yellow]Сопоставлено только {count} суставов. Проверьте XML.[/yellow]")

        self.qpos_mapping = {}
        for model_idx, mujoco_act_idx in self.actuator_mapping.items():
            joint_id = self.model.actuator_trnid[mujoco_act_idx, 0]
            qpos_idx = self.model.jnt_qposadr[joint_id]
            self.qpos_mapping[model_idx] = qpos_idx

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        return self._get_obs()
    
    def _get_obs(self):
        state = np.zeros(28, dtype=np.float32)
        for model_idx in range(28):
            if model_idx in self.qpos_mapping:
                qpos_idx = self.qpos_mapping[model_idx]
                state[model_idx] = self.data.qpos[qpos_idx]
        
        self.renderer.update_scene(self.data, camera=self.cam_id)
        image = self.renderer.render()
        return {"state": state, "image": image}
    
    def step(self, action):
        for model_idx in range(28):
            if model_idx in self.actuator_mapping:
                mujoco_act_idx = self.actuator_mapping[model_idx]
                self.data.ctrl[mujoco_act_idx] = float(action[model_idx])
        
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
    alpha = 1.85  # Очень сильное сглаживание для максимальной плавности
    
    # Параметры для фиксации частоты 30 Гц
    target_dt =0.15  # 1 действие в секунду
    
    console.print("[yellow]Запуск цикла инференса (30 Гц, максимальная плавность)...[/yellow]\n")

    for step in range(num_steps):
        start_time = time.time()
        
        # 1. Формирование наблюдений
        raw_obs = {
            "prompt": prompt,
            "observation.state": obs["state"],
            "observation.image.egocentric": obs["image"],
        }
        
        # 2. Применение входных трансформов
        transformed_obs = input_transforms(raw_obs)
        preprocessed = torch_preprocess_dict_inference(transformed_obs)
        batch = move_dict_to_batch_for_inference(preprocessed, device=device)
        
        # 3. Ручная подготовка изображения (гарантия формата B,C,H,W)
        if "observation.image.egocentric" in preprocessed:
            img_np = preprocessed["observation.image.egocentric"]
            img_tensor = torch.from_numpy(img_np).float() / 255.0
            
            if img_tensor.dim() == 3:
                h, w, c = img_tensor.shape
                if c == 3:
                    img_tensor = img_tensor.permute(2, 0, 1)
            
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            
            if img_tensor.shape[-2:] != (448, 448):
                img_tensor = F.interpolate(img_tensor, size=(448, 448), mode="bilinear", align_corners=False)
            
            if "image" not in batch:
                batch["image"] = {}
            batch["image"]["egocentric"] = img_tensor.to(device)
        
        # 4. Инференс модели
        with torch.inference_mode():
            raw_actions = policy.select_action(batch)
        
        first_step_action = raw_actions[:, 0:1, :]
        
        actions_dict = output_transforms({
            "actions": first_step_action.cpu().numpy(), 
            "state": batch["state"].cpu().numpy()
        })
        
        action = actions_dict["actions"][0, 0, :28]
        
        # 5. Сглаживание действий (очень сильное для плавности)
        if prev_action is not None:
            action = alpha * action + (1 - alpha) * prev_action
        prev_action = action.copy()

        # 6. Шаг симуляции
        obs = env.step(action)
        video_writer.append_data(obs["image"])
        
        # Логирование
        if step % 50 == 0:
            table = Table(title=f"Шаг {step}/{num_steps}")
            table.add_column("Сустав", style="cyan")
            table.add_column("Action (rad)", justify="right", style="green")
            for name, val in zip(JOINT_NAMES, action):
                table.add_row(name, f"{val:.3f}")
            console.print(table)
        
        # 7. Контроль частоты (Fix FPS to 30Hz)
        elapsed_time = time.time() - start_time
        sleep_time = target_dt - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            if step % 50 == 0:
                console.print(f"[yellow]Warning: Step {step} took {elapsed_time:.4f}s (> {target_dt:.4f}s). Cannot maintain 30Hz.[/yellow]")
    
    video_writer.close()
    console.print(f"\n[bold green]Готово: {video_path}[/bold green]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/human/Kruglov/GreenVLA/outputs/g1_r1_2b/checkpoints/last")
    parser.add_argument("--xml-path", type=str, default="/home/human/Kruglov/g1_model/unitree_g1/scene_table_object.xml")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--video-path", type=str, default="g1_output.mp4")
    parser.add_argument("--prompt", type=str, default="stack the yellow cubes")
    args = parser.parse_args()
    
    run_inference(args.model_path, args.xml_path, args.num_steps, args.video_path, args.prompt)
