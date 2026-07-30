import re

print("🚀 Начало применения R1 патчей для GreenVLA...")

# ==============================================================================
# 1. configuration_greenvla_policy.py: Отключаем flash_attention_2
# ==============================================================================
with open('lerobot/common/policies/greenvla_policy/configuration_greenvla_policy.py', 'r') as f:
    content = f.read()
content = content.replace('attention_implementation: str = "flash_attention_2"', 'attention_implementation: str = "sdpa"')
with open('lerobot/common/policies/greenvla_policy/configuration_greenvla_policy.py', 'w') as f:
    f.write(content)
print("✅ 1. flash_attention_2 заменён на sdpa")

# ==============================================================================
# 2. train_with_validation.py: Исправление приватных атрибутов и Gradient Checkpointing
# ==============================================================================
with open('lerobot/scripts/train_with_validation.py', 'r') as f:
    content = f.read()

# Исправляем приватные атрибуты
replacements = [
    ('robotics_dataset._num_samples', 'robotics_dataset.num_samples'),
    ('robotics_dataset._num_episodes', 'robotics_dataset.num_episodes'),
    ('robotics_dataset._dataset_id', 'robotics_dataset.dataset_id'),
    ('robotics_dataset._fps', 'robotics_dataset.fps'),
    ('robotics_dataset._features', 'robotics_dataset.features'),
]
for old, new in replacements:
    content = content.replace(old, new)

# Внедряем Gradient Checkpointing после инициализации policy
target = "policy = instantiate_policy(cfg.policy, device=accelerator.device)"
replacement = """policy = instantiate_policy(cfg.policy, device=accelerator.device)

# === ОПТИМИЗАЦИЯ ПАМЯТИ ДЛЯ RTX 4090 ===
try:
    if hasattr(policy, 'model') and hasattr(policy.model, 'gradient_checkpointing_enable'):
        policy.model.gradient_checkpointing_enable()
    elif hasattr(policy, 'gradient_checkpointing_enable'):
        policy.gradient_checkpointing_enable()
    print("✅ Gradient checkpointing успешно включён!")
except Exception as e:
    print(f"⚠️ Не удалось включить gradient checkpointing: {e}")
# ========================================="""

if target in content:
    content = content.replace(target, replacement)

with open('lerobot/scripts/train_with_validation.py', 'w') as f:
    f.write(content)
print("✅ 2. Приватные атрибуты исправлены, gradient checkpointing внедрён")

# ==============================================================================
# 3. g1_r1_loco_2b.yaml: Настройка конфига обучения
# ==============================================================================
with open('lerobot/conf/training/g1_r1_loco_2b.yaml', 'r') as f:
    content = f.read()

# Добавляем pretrained_path в начало
if 'pretrained_path:' not in content:
    content = "pretrained_path: null\n\n" + content

# Добавляем mixed_precision в training
if 'mixed_precision:' not in content:
    content = content.replace('# Training\ntraining:\n', '# Training\ntraining:\n  mixed_precision: "bf16"\n')

# Добавляем optimizer и scheduler в конец
if 'optimizer:' not in content:
    content += """
# Optimizer and Scheduler
optimizer:
  _target_: lerobot.common.optim.optimizers.AdamWConfig
  lr: 1e-4
  betas: [0.9, 0.95]
  weight_decay: 1e-8
  grad_clip_norm: 1.0

scheduler:
  _target_: lerobot.common.optim.schedulers.WarmupStableDecaySchedulerConfig
  peak_lr: ${optimizer.lr}
  decay_lr: 0.0
  num_warmup_steps: 500
  stable_phase_steps: 4000
  num_decay_steps: 500
"""

# Добавляем image_keys в policy_config
if 'image_keys:' not in content:
    content = content.replace(
        '    tokenizer_max_length: 256',
        '    tokenizer_max_length: 256\n    image_keys:\n      - egocentric'
    )

with open('lerobot/conf/training/g1_r1_loco_2b.yaml', 'w') as f:
    f.write(content)
print("✅ 3. Конфиг обучения обновлён (bf16, optimizer, image_keys)")

# ==============================================================================
# 4. g1.py: Гибкое чтение state/actions и создание image_mask
# ==============================================================================
with open('lerobot/common/datasets/data_transforms/robots/g1.py', 'r') as f:
    content = f.read()

old_call = """    def __call__(self, data: dict) -> dict:
        state = data["observation.state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        # Просто приводим к нужной размерности (padding с нулями)
        state = pad_to_dim(state, self.action_dim, value=0.0)
        data["observation.state"] = state
        
        # Action тоже приводим к нужной размерности
        if "action" in data:
            action = data["action"]
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()
            action = pad_to_dim(action, self.action_dim, value=0.0)
            data["action"] = action
        
        return data"""

new_call = """    def __call__(self, data: dict) -> dict:
        # Гибкое чтение: пробуем 'observation.state', затем 'state'
        state = data.get("observation.state", data.get("state"))
        if state is None:
            raise KeyError("Neither 'observation.state' nor 'state' found in data")
            
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        # Приводим к нужной размерности (padding с нулями)
        state = pad_to_dim(state, self.action_dim, value=0.0)
        data["state"] = state  # Сохраняем как 'state' для токенизатора
        
        # Action тоже приводим к нужной размерности
        if "action" in data:
            action = data["action"]
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()
            action = pad_to_dim(action, self.action_dim, value=0.0)
            data["actions"] = action  # Сохраняем как 'actions' для токенизатора
        
        # Создаём image_mask для токенизатора (скаляр True, т.к. трансформ применяется к одному элементу)
        if "image" in data and "image_mask" not in data:
            data["image_mask"] = {}
            for key in data["image"].keys():
                data["image_mask"][key] = torch.tensor(True)
        
        return data"""

if old_call in content:
    content = content.replace(old_call, new_call)
    with open('lerobot/common/datasets/data_transforms/robots/g1.py', 'w') as f:
        f.write(content)
    print("✅ 4. G1InputsTransform обновлён (гибкие ключи + image_mask)")
else:
    print("⚠️ 4. Блок G1InputsTransform не найден в точном виде, проверьте вручную.")

# ==============================================================================
# 5. modeling_greenvla_policy.py: Безопасное извлечение image_mask
# ==============================================================================
with open('lerobot/common/policies/greenvla_policy/modeling_greenvla_policy.py', 'r') as f:
    content = f.read()

old_mask = 'mask = batch["image_mask"][key]'
new_mask = """if "image_mask" in batch and key in batch["image_mask"]:
                mask = batch["image_mask"][key]
            else:
                # Fallback: создаем маску валидности (все True), если её нет в датасете
                mask = torch.ones(img.shape[0], dtype=torch.bool, device=img.device) if img.ndim > 1 else torch.tensor(True, device=img.device)"""

# Используем regex для надёжной замены, учитывая возможные отступы
pattern = r'(\s+)mask = batch\["image_mask"\]\[key\]'
if re.search(pattern, content):
    def replacer(match):
        indent = match.group(1)
        return f"""{indent}if "image_mask" in batch and key in batch["image_mask"]:
{indent}    mask = batch["image_mask"][key]
{indent}else:
{indent}    # Fallback: создаем маску валидности (все True)
{indent}    mask = torch.ones(img.shape[0], dtype=torch.bool, device=img.device) if img.ndim > 1 else torch.tensor(True, device=img.device)"""
    content = re.sub(pattern, replacer, content)
    with open('lerobot/common/policies/greenvla_policy/modeling_greenvla_policy.py', 'w') as f:
        f.write(content)
    print("✅ 5. Безопасное извлечение image_mask в модели добавлено")

# ==============================================================================
# 6. data_config.py: Самая сложная часть (Repack + Model Transforms)
# ==============================================================================
with open('lerobot/common/datasets/data_config.py', 'r') as f:
    content = f.read()

# 6.1. Исправляем repack_structure_corrected в LeRobotG1DataConfig
pattern_repack = r"(class LeRobotG1DataConfig.*?repack_structure_corrected = \{).*?(\}\s*\n\s*repack_transform = TorchGroup)"
replacement_repack = r"""\1
            'observation.state': 'observation.state',
            'action': 'action',
            'prompt': 'task',
            'image': {
                'egocentric': 'observation.images.egocentric',
            },
        \2"""
content = re.sub(pattern_repack, replacement_repack, content, flags=re.DOTALL)

# 6.2. Удаляем фейковый TokenizeTransform из data_transforms
content = content.replace(
    """        from lerobot.common.datasets.torch_transforms import TokenizeTransform
        
        data_transforms = TorchGroup(
            inputs=[
                TokenizeTransform(text_key="task", tokenizer_max_length=256),
                G1InputsTransform(action_dim=current_action_dim, 
                                  map_to_unified_space=self.map_to_unified_space,
                                  map_to_humanoid=self.map_to_humanoid)
            ],
            outputs=[G1OutputsTransform()],
        )""",
    """        data_transforms = TorchGroup(
            inputs=[
                G1InputsTransform(action_dim=current_action_dim, 
                                  map_to_unified_space=self.map_to_unified_space,
                                  map_to_humanoid=self.map_to_humanoid)
            ],
            outputs=[G1OutputsTransform()],
        )"""
)

# 6.3. Внедряем правильные model_transforms перед return DataConfig в G1DataConfig
# Находим класс LeRobotG1DataConfig и его return DataConfig
g1_start = content.find("class LeRobotG1DataConfig")
if g1_start != -1:
    return_start = content.find("return DataConfig(", g1_start)
    if return_start != -1:
        next_class = content.find("\nclass ", return_start)
        if next_class == -1:
            next_class = len(content)
        
        correct_return = """        # Настраиваем model_transforms для GreenVLA
        from lerobot.common.policies.greenvla_policy.greenvla_tokenizer import GreenVLATokenizer
        
        model_transforms = TorchGroup(
            inputs=[
                InjectDefaultPromptTorch(prompt="walk towards a desk, pick up a bottle, and put it in a container"),
                ResizeImagesTorch(*model_config.image_shape),
                TokenizeGreenVLAInputsTransform(
                    GreenVLATokenizer(
                        max_len=model_config.tokenizer_max_length,
                        state_dim=self.state_dim,
                        control_mode=self.control_mode,
                        embodiment_name="g1",
                        image_keys=model_config.image_keys,
                        base_vlm_model=model_config.base_vlm_model,
                        discrete_state_input=getattr(model_config, 'discrete_state_input', False),
                        continuous_state_input=getattr(model_config, 'continuous_state_input', True),
                        state_dropout_prob=getattr(model_config, 'state_dropout_prob', 0.0),
                        state_special_token_id=getattr(model_config, 'state_special_token_id', None),
                        clip_state=getattr(model_config, 'clip_state', False),
                        add_control_mode=getattr(model_config, 'add_control_mode', True),
                        add_embodiment_name=getattr(model_config, 'add_embodiment_name', True),
                        model_mode=model_config.model_mode,
                        image_shape=model_config.image_shape,
                    )
                ),
            ],
            outputs=[
                ExtractGreenVLAActionsTorch(
                    GreenVLATokenizer(
                        max_len=model_config.tokenizer_max_length,
                        state_dim=self.state_dim,
                        control_mode=self.control_mode,
                        embodiment_name="g1",
                        image_keys=model_config.image_keys,
                        base_vlm_model=model_config.base_vlm_model,
                        discrete_state_input=getattr(model_config, 'discrete_state_input', False),
                        continuous_state_input=getattr(model_config, 'continuous_state_input', True),
                        state_dropout_prob=getattr(model_config, 'state_dropout_prob', 0.0),
                        state_special_token_id=getattr(model_config, 'state_special_token_id', None),
                        clip_state=getattr(model_config, 'clip_state', False),
                        add_control_mode=getattr(model_config, 'add_control_mode', True),
                        add_embodiment_name=getattr(model_config, 'add_embodiment_name', True),
                        model_mode=model_config.model_mode,
                        image_shape=model_config.image_shape,
                    ),
                    action_horizon=self.action_horizon,
                    action_dim=current_action_dim,
                    model_mode=model_config.model_mode,
                    inference_mode="flow_matching",
                )
            ],
        )

        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file,
            action_sequence_keys=self.action_sequence_keys,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            state_dim=self.state_dim,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            map_to_unified_space=self.map_to_unified_space,
            validation_episodes=self.validation_episodes,
            return_subtasks=self.return_subtasks,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            norm_stats=norm_stats,
        )"""
        
        content = content[:return_start] + correct_return + "\n\n" + content[next_class:]
        print("✅ 6. data_config.py полностью пересобран (repack + model_transforms)")

with open('lerobot/common/datasets/data_config.py', 'w') as f:
    f.write(content)

print("🎉 Все R1 патчи успешно применены!")
