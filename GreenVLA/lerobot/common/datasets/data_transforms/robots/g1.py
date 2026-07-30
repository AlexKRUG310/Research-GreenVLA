"""Data transforms for G1 Humanoid robot."""

import numpy as np
import torch


def pad_to_dim(tensor: torch.Tensor, target_dim: int, value: float = 0.0) -> torch.Tensor:
    """Pad tensor to target dimension with specified value."""
    if tensor.shape[-1] >= target_dim:
        return tensor[..., :target_dim]
    
    pad_size = target_dim - tensor.shape[-1]
    padding = torch.full((*tensor.shape[:-1], pad_size), value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=-1)


class G1InputsTransform:
    """Transform for G1 robot inputs."""
    
    def __init__(self, action_dim: int = 48, map_to_unified_space: bool = False, map_to_humanoid: bool = False):
        self.action_dim = action_dim
        self.map_to_unified_space = map_to_unified_space
        self.map_to_humanoid = map_to_humanoid
    
    def __call__(self, data: dict) -> dict:
        # Гибкое чтение: пробуем 'observation.state', затем 'state'
        state = data.get("observation.state", data.get("observation/state", data.get("state")))
        if state is None:
            raise KeyError("Neither 'observation.state' nor 'state' found in data")
            
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        # Приводим к нужной размерности (padding с нулями)
        state = pad_to_dim(state, self.action_dim, value=0.0)
        data["state"] = state  # Сохраняем как 'state' для токенизатора
        
        # Action тоже приводим к нужной размерности
        action_key = "action" if "action" in data else "actions"
        if action_key in data:
            action = data[action_key]
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()
            elif not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)
            
            # Явный паддинг последней размерности до self.action_dim (48)
            current_dim = action.shape[-1]
            if current_dim < self.action_dim:
                pad_size = self.action_dim - current_dim
                padding = torch.zeros(*action.shape[:-1], pad_size, dtype=action.dtype, device=action.device)
                action = torch.cat([action, padding], dim=-1)
            elif current_dim > self.action_dim:
                action = action[..., :self.action_dim]
                
            data["actions"] = action
        return data


class G1OutputsTransform:
    """Transform for G1 robot outputs."""
    
    def __call__(self, data: dict) -> dict:
        # Просто возвращаем данные как есть
        return data
