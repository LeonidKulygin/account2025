from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from enum import Enum


class OptimizerType(str, Enum):
    """Типы оптимайзеров"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RADAM = "radam"


class SchedulerType(str, Enum):
    """Типы scheduler'ов"""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    CYCLICAL = "cyclical"
    EXPONENTIAL = "exponential"
    STEP = "step"


@dataclass
class OptimizerConfig:
    """Конфиг оптимайзера"""
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    
    # Базовые параметры
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Adam параметры
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    # AdamW параметры
    use_correct_wd: bool = True  # Правильный weight decay
    
    # SGD параметры
    momentum: float = 0.9
    nesterov: bool = True
    
    # Gradient clipping
    clip_grad_norm: Optional[float] = 1.0
    clip_grad_value: Optional[float] = None


@dataclass
class SchedulerConfig:
    """Конфиг scheduler'а"""
    scheduler_type: SchedulerType = SchedulerType.CYCLICAL
    
    # Cyclical LR
    base_lr: float = 1e-3
    max_lr: float = 1e-2
    cycle_size: int = 4  # эпохи
    
    # Cosine Annealing
    t_max: int = 50
    eta_min: float = 1e-6
    
    # Linear warmup
    warmup_steps: int = 0
    warmup_epochs: int = 2
    
    # Step LR
    step_size: int = 10
    gamma: float = 0.1
    
    # Exponential
    exp_gamma: float = 0.95


@dataclass
class RegularizationConfig:
    """Конфиг регуляризации"""
    # L1/L2
    l1_weight: float = 0.0
    l2_weight: float = 1e-4
    
    # Dropout
    dropout_rate: float = 0.3
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_gini"  # gini, auc, loss
    
    # Gradient accumulation
    accumulation_steps: int = 1


@dataclass
class LossConfig:
    """Конфиг функции потерь"""
    loss_type: str = "bce"  # bce, focal, weighted_bce
    
    # Sample weighting
    use_sample_weight: bool = True
    sample_weight_method: str = "temporal"  # temporal, class_balance
    
    # Focal loss параметры
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Class weights
    pos_weight: Optional[float] = None


@dataclass
class TrainingConfig:
    """Главная конфигурация обучения"""
    
    # Компоненты
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    # Параметры обучения
    num_epochs: int = 100
    batch_size: int = 128
    validation_batch_size: int = 256
    
    # Warmup
    warmup_epochs: int = 2
    warmup_lr_init: float = 1e-6
    
    # Смешанная точность (AMP)
    use_amp: bool = True
    
    # Чекпоинты
    save_best_only: bool = True
    save_last: bool = True
    checkpoint_metric: str = "val_gini"
    
    # Логирование
    log_every_n_steps: int = 50
    log_every_n_epochs: int = 1
    
    # Валидация
    validate_every_n_epochs: int = 1
    
    def to_dict(self) -> Dict:
        """Конвертировать в словарь"""
        return {
            'optimizer': self.optimizer.__dict__,
            'scheduler': self.scheduler.__dict__,
            'regularization': self.regularization.__dict__,
            'loss': self.loss.__dict__,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
        }
    
    @classmethod
    def get_baseline(cls):
        """Базовая конфигурация"""
        return cls(
            optimizer=OptimizerConfig(
                optimizer_type=OptimizerType.ADAMW,
                learning_rate=1e-3,
                weight_decay=1e-4,
            ),
            scheduler=SchedulerConfig(
                scheduler_type=SchedulerType.CYCLICAL,
                base_lr=1e-3,
                max_lr=1e-2,
            ),
            num_epochs=50,
            batch_size=128,
        )
    
    @classmethod
    def get_aggressive(cls):
        """Агрессивное обучение"""
        return cls(
            optimizer=OptimizerConfig(
                optimizer_type=OptimizerType.ADAM,
                learning_rate=5e-3,
                weight_decay=1e-5,
            ),
            scheduler=SchedulerConfig(
                scheduler_type=SchedulerType.COSINE,
                t_max=100,
            ),
            regularization=RegularizationConfig(
                dropout_rate=0.4,
                early_stopping_patience=5,
            ),
            num_epochs=100,
            batch_size=64,
        )
