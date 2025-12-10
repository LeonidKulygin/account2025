import torch
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def collate_variable_length(
    batch: List[Dict],
    pad_value: float = 0.0,
    pad_token: int = -1,
) -> Dict[str, torch.Tensor]:
    """
    Collate функция для переменной длины последовательностей
    
    Характеристики:
    - Динамическое padding до максимума в батче
    - Маскирование padding позиций
    - Экономия памяти GPU
    
    Args:
        batch: List of dicts от Dataset
        pad_value: Значение для padding (для float тензоров)
        pad_token: Значение для padding (для int тензоров)
    
    Returns:
        Dict с батчированными тензорами и masks
    """
    
    # Получить максимальную длину в батче
    max_length = max(
        item['transactions'].shape[0] for item in batch
    )
    
    batch_size = len(batch)
    feature_dim = batch[0]['transactions'].shape[1]
    
    # Инициализировать выходные тензоры
    transactions = torch.full(
        (batch_size, max_length, feature_dim),
        pad_value,
        dtype=torch.float32,
    )
    
    # Маска для padding позиций
    # 1 = реальные данные, 0 = padding
    mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.bool,
    )
    
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Заполнить батч
    for i, item in enumerate(batch):
        seq = item['transactions']
        seq_len = seq.shape[0]
        
        # Скопировать последовательность
        transactions[i, :seq_len] = torch.FloatTensor(seq)
        
        # Установить маску
        mask[i, :seq_len] = True
        
        # Сохранить длину
        lengths[i] = seq_len
    
    # Получить другие признаки
    num_cat_features = batch[0]['categorical_features'].shape[0]
    num_offer_features = batch[0]['offer_features'].shape[0]
    
    categorical_features = torch.zeros(
        (batch_size, num_cat_features),
        dtype=torch.long,
    )
    
    offer_features = torch.zeros(
        (batch_size, num_offer_features),
        dtype=torch.long,
    )
    
    targets = torch.zeros(batch_size, dtype=torch.float32)
    weights = torch.ones(batch_size, dtype=torch.float32)
    
    for i, item in enumerate(batch):
        categorical_features[i] = torch.LongTensor(item['categorical_features'])
        offer_features[i] = torch.LongTensor(item['offer_features'])
        targets[i] = item['targets']
        
        if 'weights' in item:
            weights[i] = item['weights']
    
    return {
        'transactions': transactions,
        'categorical_features': categorical_features,
        'offer_features': offer_features,
        'targets': targets,
        'lengths': lengths,
        'mask': mask,
        'weights': weights,
    }


def collate_fixed_length(
    batch: List[Dict],
    max_length: int = 550,
    pad_value: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Collate функция для фиксированной длины последовательности
    
    Характеристики:
    - Padding/truncating до фиксированной длины
    - Быстрее variable_length (меньше управления памятью)
    - Может быть менее эффективно по памяти
    
    Args:
        batch: List of dicts от Dataset
        max_length: Максимальная длина последовательности
        pad_value: Значение для padding
    
    Returns:
        Dict с батчированными тензорами
    """
    
    batch_size = len(batch)
    feature_dim = batch[0]['transactions'].shape[1]
    
    # Инициализировать выходные тензоры
    transactions = torch.full(
        (batch_size, max_length, feature_dim),
        pad_value,
        dtype=torch.float32,
    )
    
    mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.bool,
    )
    
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Заполнить батч
    for i, item in enumerate(batch):
        seq = torch.FloatTensor(item['transactions'])
        seq_len = min(len(seq), max_length)
        
        # Либо truncate либо использовать целиком
        if len(seq) > max_length:
            seq = seq[-max_length:]  # Взять последние max_length
        
        # Скопировать
        transactions[i, :seq_len] = seq[-seq_len:]
        
        # Маска
        mask[i, :seq_len] = True
        
        # Длина
        lengths[i] = seq_len
    
    # Другие признаки
    num_cat_features = batch[0]['categorical_features'].shape[0]
    num_offer_features = batch[0]['offer_features'].shape[0]
    
    categorical_features = torch.zeros(
        (batch_size, num_cat_features),
        dtype=torch.long,
    )
    
    offer_features = torch.zeros(
        (batch_size, num_offer_features),
        dtype=torch.long,
    )
    
    targets = torch.zeros(batch_size, dtype=torch.float32)
    weights = torch.ones(batch_size, dtype=torch.float32)
    
    for i, item in enumerate(batch):
        categorical_features[i] = torch.LongTensor(item['categorical_features'])
        offer_features[i] = torch.LongTensor(item['offer_features'])
        targets[i] = item['targets']
        
        if 'weights' in item:
            weights[i] = item['weights']
    
    return {
        'transactions': transactions,
        'categorical_features': categorical_features,
        'offer_features': offer_features,
        'targets': targets,
        'lengths': lengths,
        'mask': mask,
        'weights': weights,
    }


def collate_bucket_batch(
    batch: List[Dict],
    pad_value: float = 0.0,
    pad_token: int = -1,
) -> Dict[str, torch.Tensor]:
    """
    Collate функция с автоматическим bucketing по длине
    
    Используется когда SequenceBatchSampler уже сгруппировал по длине
    
    Характеристики:
    - Все последовательности в батче похожей длины
    - Минимум padding
    - Максимальная эффективность памяти
    
    Args:
        batch: List of dicts от Dataset (уже с похожей длиной)
        pad_value: Значение для padding
        pad_token: Значение для padding (int)
    
    Returns:
        Dict с батчированными тензорами
    """
    
    batch_size = len(batch)
    
    # Найти максимум в этом батче (будет близко к min батче)
    max_length = max(
        item['transactions'].shape[0] for item in batch
    )
    
    feature_dim = batch[0]['transactions'].shape[1]
    
    # Инициализировать
    transactions = torch.full(
        (batch_size, max_length, feature_dim),
        pad_value,
        dtype=torch.float32,
    )
    
    mask = torch.ones(
        (batch_size, max_length),
        dtype=torch.bool,
    )
    
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # Заполнить батч
    for i, item in enumerate(batch):
        seq = torch.FloatTensor(item['transactions'])
        seq_len = len(seq)
        
        transactions[i, :seq_len] = seq
        mask[i, seq_len:] = False  # Маска padding позиций
        lengths[i] = seq_len
    
    # Категориальные признаки
    num_cat_features = batch[0]['categorical_features'].shape[0]
    num_offer_features = batch[0]['offer_features'].shape[0]
    
    categorical_features = torch.zeros(
        (batch_size, num_cat_features),
        dtype=torch.long,
    )
    
    offer_features = torch.zeros(
        (batch_size, num_offer_features),
        dtype=torch.long,
    )
    
    targets = torch.zeros(batch_size, dtype=torch.float32)
    weights = torch.ones(batch_size, dtype=torch.float32)
    
    for i, item in enumerate(batch):
        categorical_features[i] = torch.LongTensor(item['categorical_features'])
        offer_features[i] = torch.LongTensor(item['offer_features'])
        targets[i] = item['targets']
        
        if 'weights' in item:
            weights[i] = item['weights']
    
    return {
        'transactions': transactions,
        'categorical_features': categorical_features,
        'offer_features': offer_features,
        'targets': targets,
        'lengths': lengths,
        'mask': mask,
        'weights': weights,
    }


class PaddedBatchCollator:
    """
    Класс для управления collate операциями с параметрами
    """
    
    def __init__(
        self,
        mode: str = 'variable',
        max_length: Optional[int] = 550,
        pad_value: float = 0.0,
        pad_token: int = -1,
    ):
        """
        Инициализация collator
        
        Args:
            mode: 'variable', 'fixed', или 'bucket'
            max_length: Максимальная длина (для 'fixed' mode)
            pad_value: Значение padding для float
            pad_token: Значение padding для int
        """
        self.mode = mode
        self.max_length = max_length
        self.pad_value = pad_value
        self.pad_token = pad_token
        
        # Выбрать функцию
        if mode == 'variable':
            self.collate_fn = collate_variable_length
        elif mode == 'fixed':
            self.collate_fn = collate_fixed_length
        elif mode == 'bucket':
            self.collate_fn = collate_bucket_batch
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Выполнить collate операцию
        
        Args:
            batch: List of dicts от Dataset
        
        Returns:
            Dict с батчированными данными
        """
        if self.mode == 'fixed':
            return self.collate_fn(
                batch,
                max_length=self.max_length,
                pad_value=self.pad_value,
            )
        else:
            return self.collate_fn(
                batch,
                pad_value=self.pad_value,
                pad_token=self.pad_token,
            )


# Вспомогательная функция для использования с DataLoader
def get_collate_fn(mode: str = 'variable', max_length: int = 550):
    """
    Получить collate функцию
    
    Пример использования:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=64,
            collate_fn=get_collate_fn('variable'),
        )
    """
    collator = PaddedBatchCollator(
        mode=mode,
        max_length=max_length,
    )
    return collator
