from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import yaml
from datetime import datetime


@dataclass
class BaseConfig:
    """Базовый класс конфигурации"""
    
    # Проект и эксперимент
    project_name: str = "credit-risk-model"
    experiment_name: str = "baseline"
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Пути
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    log_dir: Path = Path("outputs/logs")
    
    # Logging
    log_level: str = "INFO"
    use_clearml: bool = True
    use_wandb: bool = False
    clearml_project: str = "credit-risk"
    clearml_task: str = "experiment"
    
    # Device
    device: str = "cuda"
    seed: int = 42
    deterministic: bool = True
    
    # Reproducibility
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        """Создать директории если их нет"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: str):
        """Загрузить конфиг из YAML"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Конвертировать в словарь"""
        return self.__dict__.copy()
    
    def save(self, path: str):
        """Сохранить конфиг в YAML"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
