import torch
import torch.nn as nn
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Optional
import click

from src.config.model_config import ModelConfig
from src.model.factory import ModelFactory
from src.data.loader import DataLoader
from src.data.preprocessor import TextPreprocessor
from src.evaluation.metrics import compute_metrics
from src.utils import get_device

logger = logging.getLogger(__name__)


class CreditRiskInference:
    """Класс для inference модели на новых данных"""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        model_config: ModelConfig,
        device: str = "auto",
        preprocessor: Optional[TextPreprocessor] = None,
    ):
        """
        Инициализация inference класса
        
        Args:
            model_path: Путь к сохраненной модели
            model_config: Конфигурация модели
            device: Device для inference ('cuda', 'cpu', 'auto')
            preprocessor: Препроцессор текста (если None, используется дефолтный)
        """
        self.model_config = model_config
        self.device = get_device(device)
        self.preprocessor = preprocessor or TextPreprocessor()
        
        # Загрузить модель
        self.model = self._load_model(model_path)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self, model_path: Union[str, Path]) -> nn.Module:
        """Загрузить модель с весами"""
        model_path = Path(model_path)
        
        # Создать архитектуру модели
        model = ModelFactory.create(self.model_config)
        
        # Загрузить веса
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Обработать checkpoint
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            # Если это полный checkpoint (с optimizer_state и т.д.)
            model.load_state_dict(checkpoint['model_state'])
        else:
            # Если это просто state_dict
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    @torch.no_grad()
    def predict_batch(
        self,
        transaction_sequences: torch.Tensor,
        categorical_features: torch.Tensor,
        offer_features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Делать предсказание на батче
        
        Args:
            transaction_sequences: (batch_size, seq_len, feature_dim)
            categorical_features: (batch_size, num_categorical)
            offer_features: (batch_size, num_offer_features)
            lengths: (batch_size,) - длины последовательностей
        
        Returns:
            (batch_size,) - вероятности дефолта
        """
        # Переместить на device
        transaction_sequences = transaction_sequences.to(self.device)
        categorical_features = categorical_features.to(self.device)
        offer_features = offer_features.to(self.device)
        if lengths is not None:
            lengths = lengths.to(self.device)
        
        # Forward pass
        logits = self.model(
            transaction_sequences,
            categorical_features,
            offer_features,
            lengths,
        )
        
        # Сконвертировать в вероятности
        probabilities = torch.sigmoid(logits).squeeze(-1)
        
        return probabilities.cpu().numpy()
    
    def predict_dataframe(
        self,
        data: pd.DataFrame,
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Делать предсказания на DataFrame
        
        Args:
            data: DataFrame с признаками
            batch_size: Размер батча
            show_progress: Показывать progress bar
        
        Returns:
            Массив вероятностей дефолта
        """
        predictions = []
        
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        iterator = range(num_batches)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Predicting")
        
        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(data))
            
            batch = data.iloc[start_idx:end_idx]
            
            # TODO: Обработать батч в нужный формат
            # batch_dict = self._prepare_batch(batch)
            
            # Делать предсказание
            # batch_pred = self.predict_batch(**batch_dict)
            # predictions.append(batch_pred)
        
        return np.concatenate(predictions) if predictions else np.array([])
    
    def predict_with_explanations(
        self,
        transaction_sequences: torch.Tensor,
        categorical_features: torch.Tensor,
        offer_features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Делать предсказания с объяснениями (feature importance)
        
        Args:
            transaction_sequences: Транзакционные последовательности
            categorical_features: Категориальные признаки
            offer_features: Признаки предложения
            lengths: Длины последовательностей
        
        Returns:
            Словарь с предсказаниями и важностью признаков
        """
        # Получить предсказания
        predictions = self.predict_batch(
            transaction_sequences,
            categorical_features,
            offer_features,
            lengths,
        )
        
        # TODO: Реализовать SHAP или LIME для объяснений
        
        return {
            'predictions': predictions,
            'feature_importance': {},  # TODO
        }
    
    def batch_evaluate(
        self,
        data_loader,
        metric_names: Optional[list] = None,
    ) -> Dict:
        """
        Оценить модель на батчах
        
        Args:
            data_loader: DataLoader с данными
            metric_names: Какие метрики вычислить
        
        Returns:
            Словарь метрик
        """
        all_predictions = []
        all_targets = []
        
        for batch in data_loader:
            # TODO: Процесс батча
            # predictions = self.predict_batch(...)
            # all_predictions.append(predictions)
            # all_targets.append(batch['targets'])
        
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        
        # Вычислить метрики
        metrics = compute_metrics(predictions, targets)
        
        return metrics


@click.command()
@click.option('--model-path', type=str, required=True, 
              help='Путь к сохраненной модели')
@click.option('--config', type=str, required=True,
              help='Путь к конфиг файлу модели')
@click.option('--data-path', type=str, required=True,
              help='Путь к данным для inference')
@click.option('--output-path', type=str, required=True,
              help='Путь для сохранения предсказаний')
@click.option('--batch-size', type=int, default=256,
              help='Размер батча')
@click.option('--device', type=str, default='auto',
              help='Device: cuda, cpu, auto')
def main(
    model_path: str,
    config: str,
    data_path: str,
    output_path: str,
    batch_size: int,
    device: str,
):
    """
    Основной скрипт для inference
    
    Пример:
        python inference.py \\
            --model-path outputs/checkpoints/best.pth \\
            --config src/config/experiments/baseline.yaml \\
            --data-path data/test.csv \\
            --output-path predictions.csv
    """
    import yaml
    
    logger.basicConfig(level=logging.INFO)
    
    # Загрузить конфиг
    with open(config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    model_config = ModelConfig(**config_dict)
    
    # Создать inference
    inference = CreditRiskInference(
        model_path=model_path,
        model_config=model_config,
        device=device,
    )
    
    # Загрузить данные
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Делать предсказания
    logger.info(f"Making predictions on {len(data)} samples")
    predictions = inference.predict_dataframe(data, batch_size=batch_size)
    
    # Сохранить результаты
    results = pd.DataFrame({
        'id': data.index,
        'prediction': predictions,
        'default_probability': predictions,
    })
    
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    # Статистика
    logger.info(f"Mean prediction: {predictions.mean():.4f}")
    logger.info(f"Std prediction: {predictions.std():.4f}")
    logger.info(f"Min prediction: {predictions.min():.4f}")
    logger.info(f"Max prediction: {predictions.max():.4f}")


if __name__ == '__main__':
    main()
