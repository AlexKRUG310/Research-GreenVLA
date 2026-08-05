#!/usr/bin/env python3
"""
Инференс GreenVLA в MuJoCo для Unitree G1 с подробным выводом действий по суставам.
"""
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
import torchvision.transforms as T

console = Console()

# Карта имен суставов для 28 действий модели
JOINT_NAMES = [
    # 0-6: Левая кисть
    "L_Thumb_0", "L_Thumb_1", "L_Thumb_2", "L_Middle_0", "L_Middle_1", "L_Index_0", "L_Index_1",
    # 7-13: Правая кисть
    "R_Thumb_0", "R_Thumb_1", "R_Thumb_2", "R_Index_0", "R_Index_1", "R_Middle_0", "R_Middle_1",
    # 14-20: Левая рука
    "L_Sh_Pitch", "L_Sh_Roll", "L_Sh_Yaw", "L_Elbow", "L_Wr_Roll", "L_Wr_Pitch", "L_Wr_Yaw",
    # 21-27: Правая рука
    "R_Sh_Pitch", "R_Sh_Roll", "R_Sh_Yaw", "R_Elbow", "R_Wr_Roll", "R_Wr_Pitch", "R_Wr_Yaw"
]

class G1MuJoCoEnv:
    def __init__(self, xml_path: str, render_width=640, render_height=480):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, render_height, render_width)
        
        print(f"\n Камеры в сцене:")
        for i in range(self.model.ncam):
            cam_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            print(f"  [{i}] {cam_name}")
        
        self.actuator_mapping = {
            0: 22, 1: 23, 2: 24, 3: 25, 4: 26, 5: 27, 6: 28,
            7: 36, 8: 37, 9: 38, 10: 39, 11: 40, 12: 41, 13: 42,
            14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21,
            21: 29, 22: 30, 23: 31, 24: 32, 25: 33, 26: 34, 27: 35,
        }
        
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
            qpos_idx = self.qpos_mapping[model_idx]
            state[model_idx] = self.data.qpos[qpos_idx]
        
        self.renderer.update_scene(self.data, camera="head_camera")
        image = self.renderer.render()
        return {"state": state, "image": image}
    
    def step(self, action):
        for model_idx in range(28):
            mujoco_act_idx = self.actuator_mapping[model_idx]
            self.data.ctrl[mujoco_act_idx] = action[model_idx]
        
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        return self._get_obs()

def run_inference(model_path, xml_path, num_steps=200, video_path="out.mp4", prompt="pick up cube"):
    device = "cuda"
    console.print(f"\n[bold cyan] Загрузка модели...[/bold cyan]")
    
    policy, input_transforms, output_transforms = load_pretrained_policy(
        model_path, data_config_name="g1_r1_loco_2b", config_overrides={"device": device}
    )
    policy.to(device).eval()
    
    console.print(f"[bold cyan] Инициализация MuJoCo...[/bold cyan]")
    env = G1MuJoCoEnv(xml_path)
    obs = env.reset()
    
    console.print(f"[bold] Промпт: {prompt}[/bold]\n")
    
    video_writer = imageio.get_writer(video_path, fps=30)
    
    for step in range(num_steps):
        dummy_obs = {
            "prompt": prompt,
            "observation.state": obs["state"],
            "observation.image.egocentric": obs["image"],
            "image_mask": {"egocentric": True},
        }
        
        transformed_obs = input_transforms(dummy_obs)
        preprocessed = torch_preprocess_dict_inference(transformed_obs)
        batch = move_dict_to_batch_for_inference(preprocessed, device=device)
        
        if "image" not in batch: batch["image"] = {}
        img_tensor = batch["observation.image.egocentric"].to(torch.float32) / 255.0
        if img_tensor.dim() == 4 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(0, 3, 1, 2)
        img_tensor = T.Resize((448, 448), antialias=True)(img_tensor)
        batch["image"]["egocentric"] = img_tensor
        
        if "image_mask" in batch and isinstance(batch["image_mask"], dict):
            for k, v in batch["image_mask"].items():
                batch["image_mask"][k] = torch.tensor([True], dtype=torch.bool, device=device)
        
        with torch.inference_mode():
            raw_actions = policy.select_action(batch)
        
        actions_dict = output_transforms({"actions": raw_actions.cpu().numpy(), "state": batch["state"].cpu().numpy()})
        action = actions_dict["actions"][0, 0, :28]
        
        obs = env.step(action)
        video_writer.append_data(obs["image"])
        
        # Выводим таблицу действий каждые 20 шагов
        if step % 20 == 0:
            table = Table(title=f"Действия модели (Шаг {step}/{num_steps})")
            table.add_column("Сустав", justify="left", style="cyan", no_wrap=True)
            table.add_column("Значение (рад)", justify="right", style="green")
            
            for name, val in zip(JOINT_NAMES, action):
                table.add_row(name, f"{val:7.3f}")
                
            console.print(table)
    
    video_writer.close()
    console.print(f"\n[bold green]✅ Видео сохранено: {video_path}[/bold green]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/human/Kruglov/GreenVLA/outputs/g1_r1_loco_2b/checkpoints/last")
    parser.add_argument("--xml-path", type=str, default="/home/human/Kruglov/g1_madel/unitree_g1/scene_table_object.xml")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--video-path", type=str, default="/home/human/Kruglov/g1_pickup_smooth.mp4")
    parser.add_argument("--prompt", type=str, default="pick up cube")
    args = parser.parse_args()
    
    run_inference(args.model_path, args.xml_path, args.num_steps, args.video_path, args.prompt)
