#!/usr/bin/env python3
"""
Инференс GreenVLA в MuJoCo для Unitree G1 с руками.
"""
import os
import numpy as np
import torch
import mujoco
import imageio
from pathlib import Path
from rich.console import Console
from lerobot.common.policies.factory import load_pretrained_policy
from lerobot.common.utils.torch_observation import (
    move_dict_to_batch_for_inference,
    torch_preprocess_dict_inference,
)
import torchvision.transforms as T

console = Console()

class G1MuJoCoEnv:
    """Среда MuJoCo для Unitree G1 с руками."""
    
    def __init__(self, xml_path: str, render_width=640, render_height=480):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, render_height, render_width)
        
        # Маппинг: индексы модели -> индексы актуаторов MuJoCo
        # Модель выдает: [14 кистей, 14 рук]
        self.actuator_mapping = {
            # Кисти (0-13 в модели)
            0: 22, 1: 23, 2: 24, 3: 25, 4: 26, 5: 27, 6: 28,  # Левая кисть
            7: 36, 8: 37, 9: 38, 10: 39, 11: 40, 12: 41, 13: 42,  # Правая кисть
            # Руки (14-27 в модели)
            14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21,  # Левая рука
            21: 29, 22: 30, 23: 31, 24: 32, 25: 33, 26: 34, 27: 35,  # Правая рука
        }
        
    def reset(self):
        """Сброс среды."""
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()
    
    def _get_obs(self):
        """Получить наблюдение из MuJoCo."""
        # Извлекаем углы суставов для рук и кистей
        qpos = self.data.qpos.copy()
        
        # Формируем observation.state (28 значений)
        # Порядок: [14 кистей, 14 рук]
        state = np.zeros(28, dtype=np.float32)
        
        # Кисти (индексы qpos для рук: 15-28 для левой, 29-42 для правой)
        # Но в qpos могут быть в другом порядке, нужно проверить
        # Для простоты берем первые 28 значений qpos (это может быть неверно!)
        # TODO: Нужно точно определить, какие индексы qpos соответствуют рукам/кистям
        
        # Временное решение: берем индексы актуаторов
        for model_idx in range(28):
            mujoco_act_idx = self.actuator_mapping[model_idx]
            # Получаем соответствующий joint angle
            joint_id = self.model.actuator_trnid[mujoco_act_idx, 0]
            qpos_idx = self.model.jnt_qposadr[joint_id]
            state[model_idx] = self.data.qpos[qpos_idx]
        
        # Рендерим изображение
        self.renderer.update_scene(self.data)
        image = self.renderer.render()
        
        return {
            "state": state,
            "image": image,
        }
    
    def step(self, action):
        """Выполнить действие (28 значений)."""
        # Применяем к актуаторам через маппинг
        for model_idx in range(28):
            mujoco_act_idx = self.actuator_mapping[model_idx]
            self.data.ctrl[mujoco_act_idx] = action[model_idx]
        
        # Симулируем несколько шагов для стабильности
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        return self._get_obs()
    
    def render(self):
        """Рендер кадра."""
        self.renderer.update_scene(self.data)
        return self.renderer.render()


def run_inference(model_path: str, xml_path: str, num_steps: int = 100, video_path: str = "inference_video.mp4"):
    """Запуск инференса с сохранением видео."""
    
    device = "cuda"
    
    console.print(f"\n[bold cyan]🚀 Загрузка модели: {model_path}[/bold cyan]")
    
    # Загружаем модель
    policy, input_transforms, output_transforms = load_pretrained_policy(
        model_path,
        data_config_name="g1_r1_loco_2b",
        config_overrides={"device": device},
    )
    policy.to(device).eval()
    console.print("[green]✓ Модель загружена![/green]\n")
    
    # Создаем среду MuJoCo
    console.print(f"[bold cyan]🤖 Инициализация MuJoCo: {xml_path}[/bold cyan]")
    env = G1MuJoCoEnv(xml_path)
    console.print("[green]✓ Среда создана![/green]\n")
    
    # Сброс среды
    obs = env.reset()
    
    # Видео writer
    video_writer = imageio.get_writer(video_path, fps=30)
    
    console.print(f"[bold]Запуск инференса на {num_steps} шагов...[/bold]")
    
    for step in range(num_steps):
        # Подготавливаем observation для модели
        dummy_obs = {
            "prompt": "pick up the object",
            "observation.state": obs["state"],
            "observation.image.egocentric": obs["image"],
            "image_mask": {"egocentric": True},
        }
        
        # Трансформы
        transformed_obs = input_transforms(dummy_obs)
        preprocessed = torch_preprocess_dict_inference(transformed_obs)
        batch = move_dict_to_batch_for_inference(preprocessed, device=device)
        
        # Обработка изображения
        if "image" not in batch:
            batch["image"] = {}
        
        img_tensor = batch["observation.image.egocentric"].to(torch.float32) / 255.0
        if img_tensor.dim() == 4 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(0, 3, 1, 2)
        
        resize_transform = T.Resize((448, 448), antialias=True)
        img_tensor = resize_transform(img_tensor)
        batch["image"]["egocentric"] = img_tensor
        
        if "image_mask" in batch and isinstance(batch["image_mask"], dict):
            for k, v in batch["image_mask"].items():
                batch["image_mask"][k] = torch.tensor([True], dtype=torch.bool, device=device)
        
        # Инференс
        with torch.inference_mode():
            raw_actions = policy.select_action(batch)
        
        actions_dict = output_transforms({
            "actions": raw_actions.cpu().numpy(),
            "state": batch["state"].cpu().numpy(),
        })
        final_actions = actions_dict["actions"]
        
        # Берем первое действие из horizon
        action = final_actions[0, 0, :28]
        
        # Применяем действие
        obs = env.step(action)
        
        # Рендерим и сохраняем кадр
        frame = env.render()
        video_writer.append_data(frame)
        
        if step % 10 == 0:
            console.print(f"  Шаг {step}/{num_steps}")
    
    video_writer.close()
    console.print(f"\n[bold green]✅ Видео сохранено: {video_path}[/bold green]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/human/Kruglov/GreenVLA/outputs/g1_r1_loco_2b/checkpoints/last")
    parser.add_argument("--xml-path", type=str, default="/home/human/Kruglov/mujoco_menagerie/unitree_g1/scene_with_hands.xml")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--video-path", type=str, default="/home/human/Kruglov/g1_inference.mp4")
    args = parser.parse_args()
    
    run_inference(args.model_path, args.xml_path, args.num_steps, args.video_path)
