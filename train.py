import torch
import torch.nn as nn
import logging
from pathlib import Path
import click
import yaml
from typing import Dict, Optional

from src.config.base_config import BaseConfig
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.model.factory import ModelFactory
from src.utils.logger import setup_logger
from src.utils.device import get_device, set_seed
from src.utils.clearml_utils import ClearMLLogger, MetricsLogger

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Пайплайн обучения модели"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        base_config: BaseConfig,
    ):
        """
        Инициализация пайплайна
        
        Args:
            model_config: Конфигурация модели
            training_config: Конфигурация обучения
            base_config: Базовая конфигурация
        """
        self.model_config = model_config
        self.training_config = training_config
        self.base_config = base_config
        
        # Инициализировать логгеры
        self.logger = self._setup_loggers()
        
        # Инициализировать device
        set_seed(base_config.seed)
        self.device = get_device(base_config.device)
        
        # Создать модель
        self.model = self._create_model()
        
        # Создать оптимайзер и scheduler
        self.optimizer, self.scheduler = self._create_optimizer_scheduler()
    
    def _setup_loggers(self):
        """Инициализировать логгеры"""
        setup_logger(
            'train',
            self.base_config.log_dir / 'train.log',
            level=self.base_config.log_level,
        )
        
        # ClearML логгер
        if self.base_config.use_clearml:
            self.clearml_logger = ClearMLLogger(
                project_name=self.base_config.clearml_project,
                task_name=f"{self.base_config.experiment_name}_{self.base_config.experiment_id}",
            )
        else:
            self.clearml_logger = None
        
        # Метрики логгер
        self.metrics_logger = MetricsLogger(self.base_config.log_dir)
        
        return logging.getLogger(__name__)
    
    def _create_model(self) -> nn.Module:
        """Создать модель"""
        logger.info("Creating model")
        model = ModelFactory.create(self.model_config)
        model.to(self.device)
        
        # Логировать конфиг
        if self.clearml_logger:
            self.clearml_logger.log_config(self.model_config.to_dict())
        
        return model
    
    def _create_optimizer_scheduler(self):
        """Создать оптимайзер и scheduler"""
        from src.training.optimizer_factory import create_optimizer
        from src.training.scheduler_factory import create_scheduler
        
        optimizer = create_optimizer(
            self.model.parameters(),
            self.training_config.optimizer,
        )
        
        scheduler = create_scheduler(
            optimizer,
            self.training_config.scheduler,
            num_epochs=self.training_config.num_epochs,
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, train_loader, epoch: int):
        """
        Обучить одну эпоху
        
        Args:
            train_loader: DataLoader для обучения
            epoch: Номер эпохи
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Переместить батч на device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(
                batch['transactions'],
                batch['categorical_features'],
                batch['offer_features'],
                batch.get('lengths', None),
            )
            
            # Вычислить loss
            loss = self._compute_loss(predictions, batch['targets'], batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.training_config.optimizer.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.optimizer.clip_grad_norm,
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Логирование
            if (batch_idx + 1) % self.training_config.log_every_n_steps == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(
                    f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, epoch: int) -> Dict:
        """
        Валидация
        
        Args:
            val_loader: DataLoader для валидации
            epoch: Номер эпохи
        
        Returns:
            Словарь с метриками
        """
        self.model.eval()
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Переместить батч на device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Inference
                pred = self.model(
                    batch['transactions'],
                    batch['categorical_features'],
                    batch['offer_features'],
                    batch.get('lengths', None),
                )
                
                predictions.append(pred.cpu().numpy())
                targets.append(batch['targets'].cpu().numpy())
        
        import numpy as np
        from src.evaluation.metrics import compute_metrics
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        metrics = compute_metrics(predictions, targets)
        
       
        logger.info(f"Validation metrics (epoch {epoch}): {metrics}")
        
        if self.clearml_logger:
            metrics_with_prefix = {f"val/{k}": v for k, v in metrics.items()}
            self.clearml_logger.log_metrics(metrics_with_prefix, step=epoch)
        
        self.metrics_logger.log_metrics(metrics, stage='val', epoch=epoch)
        
        return metrics
    
    def _compute_loss(self, predictions, targets, batch):
        """Вычислить loss"""
     
        loss = nn.BCEWithLogitsLoss(reduction='mean')(predictions, targets.float().unsqueeze(-1))
     
        if self.training_config.loss.use_sample_weight and 'weights' in batch:
            loss = (loss * batch['weights']).mean()
        
        return loss
    
    def train(self, train_loader, val_loader):
        """
        Основной цикл обучения
        
        Args:
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
        """
        logger.info("Starting training")
        
        best_val_metric = float('-inf')
        patience_counter = 0
        
        for epoch in range(self.training_config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.training_config.num_epochs}")
            
            # Обучить эпоху
            train_loss = self.train_epoch(train_loader, epoch + 1)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Валидация
            if (epoch + 1) % self.training_config.validate_every_n_epochs == 0:
                val_metrics = self.validate(val_loader, epoch + 1)
                
                # Метрика для early stopping
                metric_value = val_metrics.get(
                    self.training_config.regularization.early_stopping_metric,
                    val_metrics.get('gini', 0),
                )
                
                # Early stopping
                if metric_value > best_val_metric:
                    best_val_metric = metric_value
                    patience_counter = 0
                    
                    # Сохранить лучшую модель
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
                else:
                    patience_counter += 1
                
                if patience_counter >= self.training_config.regularization.early_stopping_patience:
                    logger.info(f"Early stopping triggered (patience: {patience_counter})")
                    break
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
        
        logger.info("Training completed")
        
        # Закрыть ClearML логгер
        if self.clearml_logger:
            self.clearml_logger.close_task()
    
    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Сохранить чекпоинт"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        # Сохранить последний чекпоинт
        if self.training_config.save_last:
            path = self.base_config.checkpoint_dir / 'last.pth'
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint: {path}")
        
        # Сохранить лучший чекпоинт
        if is_best:
            path = self.base_config.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, path)
            logger.info(f"Saved best checkpoint: {path}")


@click.command()
@click.option('--config', type=str, required=True, help='Путь к конфиг файлу')
@click.option('--train-data', type=str, required=True, help='Путь к train данным')
@click.option('--val-data', type=str, required=True, help='Путь к val данным')
def main(config: str, train_data: str, val_data: str):
    """
    Основной скрипт для обучения
    
    Пример:
        python train.py --config src/config/experiments/baseline.yaml --train-data data/train.pq --val-data data/val.pq
    """
    # Загрузить конфиги
    with open(config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Создать конфиги
    base_config = BaseConfig()
    model_config = ModelConfig(**config_dict)
    training_config = TrainingConfig(**config_dict)
    
    # Создать пайплайн
    pipeline = TrainingPipeline(model_config, training_config, base_config)
    
    # TODO: Загрузить данные
    # train_loader = load_data(train_data, training_config.batch_size)
    # val_loader = load_data(val_data, training_config.validation_batch_size)
    
    # Обучить модель
    # pipeline.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
