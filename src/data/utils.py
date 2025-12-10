import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def get_sample_weights(
    df: pd.DataFrame,
    target_column: str = 'target',
    date_column: Optional[str] = None,
    method: str = 'temporal',
) -> np.ndarray:
    """
    Вычислить веса для примеров
    
    Args:
        df: DataFrame с данными
        target_column: Колонка с таргетом
        date_column: Колонка с датой (для temporal weights)
        method: 'temporal', 'balanced', 'combined'
    
    Returns:
        Массив весов длины len(df)
    """
    
    if method == 'temporal':
        if date_column is None:
            raise ValueError("date_column required for temporal weights")
        
        dates = pd.to_datetime(df[date_column])
        max_date = dates.max()
        days_diff = (max_date - dates).dt.days
        
        # Экспоненциальный спад
        weights = np.exp(-days_diff / 365.0) + 0.5
        
    elif method == 'balanced':
        # Веса обратные частоте класса
        class_counts = df[target_column].value_counts()
        weights = 1.0 / df[target_column].map(class_counts)
        weights = weights.values
        
    elif method == 'combined':
        if date_column is None:
            raise ValueError("date_column required for combined weights")
        
        # Комбинировать temporal и balanced
        dates = pd.to_datetime(df[date_column])
        max_date = dates.max()
        days_diff = (max_date - dates).dt.days
        temporal_w = np.exp(-days_diff / 365.0) + 0.5
        
        class_counts = df[target_column].value_counts()
        balanced_w = 1.0 / df[target_column].map(class_counts)
        
        # Нормализировать
        temporal_w = temporal_w / temporal_w.mean()
        balanced_w = balanced_w / balanced_w.mean()
        
        # Комбинировать
        weights = 0.7 * temporal_w.values + 0.3 * balanced_w.values
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Нормализировать
    weights = weights / weights.mean() * len(weights)
    
    return weights.astype(np.float32)


def create_amount_buckets(
    df: pd.DataFrame,
    amount_column: str = 'amount',
    num_buckets: int = 10,
) -> np.ndarray:
    """
    Создать бакеты для суммы транзакции
    
    Args:
        df: DataFrame с данными
        amount_column: Колонка с суммой
        num_buckets: Количество бакетов
    
    Returns:
        Массив границ бакетов
    """
    
    amounts = df[amount_column].values
    buckets = np.percentile(amounts, np.linspace(0, 100, num_buckets + 1))
    
    logger.info(f"Created {num_buckets} amount buckets")
    logger.info(f"Boundaries: {buckets}")
    
    return buckets


def create_time_buckets(
    df: pd.DataFrame,
    date_column: str = 'date',
    reference_date: Optional[pd.Timestamp] = None,
    num_buckets: int = 10,
) -> np.ndarray:
    """
    Создать бакеты для дней до скоринга
    
    Args:
        df: DataFrame с данными
        date_column: Колонка с датой
        reference_date: Дата скоринга (если None, используется max дата)
        num_buckets: Количество бакетов
    
    Returns:
        Массив границ бакетов
    """
    
    dates = pd.to_datetime(df[date_column])
    
    if reference_date is None:
        reference_date = dates.max()
    else:
        reference_date = pd.to_datetime(reference_date)
    
    days_diff = (reference_date - dates).dt.days
    buckets = np.percentile(days_diff, np.linspace(0, 100, num_buckets + 1))
    
    logger.info(f"Created {num_buckets} time buckets")
    logger.info(f"Boundaries: {buckets}")
    
    return buckets


def balance_dataset(
    df: pd.DataFrame,
    target_column: str = 'target',
    method: str = 'oversample',
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Балансировать датасет
    
    Args:
        df: DataFrame с данными
        target_column: Колонка с таргетом
        method: 'oversample' или 'undersample'
        random_state: Seed для randomness
    
    Returns:
        Сбалансированный DataFrame
    """
    
    from sklearn.utils import resample
    
    # Отделить классы
    negative = df[df[target_column] == 0]
    positive = df[df[target_column] == 1]
    
    logger.info(f"Original distribution: {len(negative)} negative, {len(positive)} positive")
    
    if method == 'oversample':
        # Oversample меньший класс
        if len(positive) < len(negative):
            positive = resample(
                positive,
                n_samples=len(negative),
                random_state=random_state,
            )
        else:
            negative = resample(
                negative,
                n_samples=len(positive),
                random_state=random_state,
            )
    
    elif method == 'undersample':
        # Undersample больший класс
        if len(positive) < len(negative):
            negative = resample(
                negative,
                n_samples=len(positive),
                random_state=random_state,
            )
        else:
            positive = resample(
                positive,
                n_samples=len(negative),
                random_state=random_state,
            )
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Объединить
    result = pd.concat([negative, positive], ignore_index=True)
    
    logger.info(f"Balanced distribution: {len(negative)} negative, {len(positive)} positive")
    
    return result


def compute_statistics(
    df: pd.DataFrame,
    numeric_columns: Optional[list] = None,
) -> Dict:
    """
    Вычислить статистику по датасету
    
    Args:
        df: DataFrame с данными
        numeric_columns: Какие колонки анализировать
    
    Returns:
        Dict с статистикой
    """
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    stats = {}
    
    for col in numeric_columns:
        if col in df.columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75)),
            }
    
    logger.info(f"Computed statistics for {len(stats)} columns")
    
    return stats


def check_data_quality(
    df: pd.DataFrame,
    verbose: bool = True,
) -> Dict:
    """
    Проверить качество данных
    
    Args:
        df: DataFrame с данными
        verbose: Выводить ли информацию
    
    Returns:
        Dict с результатами проверки
    """
    
    results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.astype(str).to_dict(),
    }
    
    if verbose:
        logger.info(f"Data Quality Check:")
        logger.info(f"  Total rows: {results['total_rows']}")
        logger.info(f"  Total columns: {results['total_columns']}")
        logger.info(f"  Duplicate rows: {results['duplicate_rows']}")
        
        if any(v > 0 for v in results['missing_values'].values()):
            logger.warning("Missing values found:")
            for col, count in results['missing_values'].items():
                if count > 0:
                    logger.warning(f"  {col}: {count} ({100*count/len(df):.1f}%)")
    
    return results


def split_by_date(
    df: pd.DataFrame,
    date_column: str = 'date',
    split_date: pd.Timestamp = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Разделить данные по дате
    
    Args:
        df: DataFrame с данными
        date_column: Колонка с датой
        split_date: Дата разделения (если None, используется медиана)
    
    Returns:
        (before, after) DataFrames
    """
    
    dates = pd.to_datetime(df[date_column])
    
    if split_date is None:
        split_date = dates.median()
    else:
        split_date = pd.to_datetime(split_date)
    
    before = df[dates < split_date]
    after = df[dates >= split_date]
    
    logger.info(f"Split by date {split_date}: {len(before)} before, {len(after)} after")
    
    return before, after


class DataNormalizer:
    """
    Нормализация признаков
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data: np.ndarray):
        """
        Обучить нормализатор
        
        Args:
            data: (N, D) массив
        """
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        
        # Избежать деления на 0
        self.std[self.std == 0] = 1.0
        
        logger.info(f"Fitted normalizer with mean shape {self.mean.shape}")
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Нормализовать данные
        
        Args:
            data: (N, D) массив
        
        Returns:
            Нормализованный массив
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        return (data - self.mean) / self.std
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Обучить и трансформировать в один шаг
        """
        self.fit(data)
        return self.transform(data)
