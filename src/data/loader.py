import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataLoader:
    """Загрузчик данных из различных форматов"""
    
    @staticmethod
    def load_csv(
        filepath: str,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Загрузить CSV файл
        
        Args:
            filepath: Путь к CSV файлу
            nrows: Количество строк для загрузки (None = все)
        
        Returns:
            DataFrame с данными
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading CSV from {filepath}")
        
        df = pd.read_csv(filepath, nrows=nrows)
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    @staticmethod
    def load_parquet(
        filepath: str,
        columns: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Загрузить Parquet файл
        
        Args:
            filepath: Путь к Parquet файлу
            columns: Какие колонки загрузить
        
        Returns:
            DataFrame с данными
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading Parquet from {filepath}")
        
        df = pd.read_parquet(filepath, columns=columns)
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    @staticmethod
    def load_multiple_files(
        directory: str,
        pattern: str = "*.csv",
    ) -> pd.DataFrame:
        """
        Загрузить и объединить несколько файлов
        
        Args:
            directory: Директория с файлами
            pattern: Паттерн для поиска файлов (например "*.csv")
        
        Returns:
            Объединенный DataFrame
        """
        directory = Path(directory)
        
        files = list(directory.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No files found matching {pattern} in {directory}")
        
        logger.info(f"Found {len(files)} files matching {pattern}")
        
        dfs = []
        for filepath in files:
            logger.info(f"Loading {filepath.name}")
            if filepath.suffix == '.csv':
                df = pd.read_csv(filepath)
            elif filepath.suffix == '.parquet':
                df = pd.read_parquet(filepath)
            else:
                continue
            
            dfs.append(df)
        
        result = pd.concat(dfs, ignore_index=True)
        
        logger.info(f"Concatenated into {len(result)} rows")
        
        return result


class TransactionDataProcessor:
    """
    Обработка и подготовка транзакционных данных
    """
    
    def __init__(
        self,
        max_seq_length: int = 550,
        min_transactions: int = 9,
    ):
        """
        Инициализация процессора
        
        Args:
            max_seq_length: Максимальная длина последовательности
            min_transactions: Минимальное количество транзакций для клиента
        """
        self.max_seq_length = max_seq_length
        self.min_transactions = min_transactions
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        client_column: str = 'client_id',
        amount_column: str = 'amount',
        description_column: str = 'description',
        send_flag_column: str = 'send_flag',
        target_column: str = 'target',
    ) -> pd.DataFrame:
        """
        Подготовить данные для модели
        
        Args:
            df: Исходный DataFrame
            date_column: Название колонки с датой
            client_column: Название колонки с ID клиента
            amount_column: Название колонки с суммой
            description_column: Название колонки с описанием
            send_flag_column: Название колонки с направлением (in/out)
            target_column: Название колонки с таргетом
        
        Returns:
            Обработанный DataFrame
        """
        logger.info("Preparing data...")
        
        # Конвертировать дату
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Сортировать по клиенту и дате
        df = df.sort_values([client_column, date_column])
        
        # Фильтровать по минимальному количеству транзакций
        client_counts = df.groupby(client_column).size()
        valid_clients = client_counts[client_counts >= self.min_transactions].index
        
        df = df[df[client_column].isin(valid_clients)].copy()
        
        logger.info(f"Filtered to {len(valid_clients)} clients with >= {self.min_transactions} transactions")
        logger.info(f"Remaining rows: {len(df)}")
        
        # Группировать по клиентам и создать последовательности
        grouped = df.groupby(client_column)
        
        processed_rows = []
        
        for client_id, group in grouped:
            # Сортировать группу по дате
            group = group.sort_values(date_column)
            
            # Взять последние max_seq_length транзакций
            if len(group) > self.max_seq_length:
                group = group.tail(self.max_seq_length)
            
            # Создать список транзакций
            transactions = []
            for _, trans_row in group.iterrows():
                transaction = {
                    'date': trans_row[date_column],
                    'description': str(trans_row.get(description_column, '')).lower(),
                    'amount': float(trans_row[amount_column]),
                    'send_flag': str(trans_row.get(send_flag_column, 'unknown')),
                }
                transactions.append(transaction)
            
            # Создать строку для датасета
            processed_row = {
                'client_id': client_id,
                'transactions': transactions,
                'target': float(group[target_column].iloc[-1]),  # Последний таргет
                'seq_length': len(transactions),
            }
            
            processed_rows.append(processed_row)
        
        result_df = pd.DataFrame(processed_rows)
        
        logger.info(f"Processed {len(result_df)} sequences")
        
        return result_df
    
    def create_buckets(
        self,
        df: pd.DataFrame,
        num_buckets: int = 10,
    ) -> Dict[str, np.ndarray]:
        """
        Создать бакеты для бакетизации количественных признаков
        
        Args:
            df: DataFrame с данными
            num_buckets: Количество бакетов
        
        Returns:
            Dict с boundaries для каждого признака
        """
        logger.info(f"Creating {num_buckets} buckets...")
        
        buckets = {}
        
        # Бакеты для длины последовательности
        seq_lengths = df['seq_length'].values
        buckets['seq_length'] = np.percentile(seq_lengths, np.linspace(0, 100, num_buckets + 1))
        
        logger.info(f"Seq length buckets: {buckets['seq_length']}")
        
        return buckets
    
    def apply_buckets(
        self,
        df: pd.DataFrame,
        buckets: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Применить бакетизацию к данным
        
        Args:
            df: DataFrame с данными
            buckets: Dict с boundaries для бакетизации
        
        Returns:
            DataFrame с добавленными бакетированными колонками
        """
        df = df.copy()
        
        for feature, boundaries in buckets.items():
            if feature in df.columns:
                bucket_col = f"{feature}_bucket"
                df[bucket_col] = np.searchsorted(boundaries[1:-1], df[feature])
                logger.info(f"Created {bucket_col}")
        
        return df


class DataSplitter:
    """
    Разделение данных на train/val/test
    """
    
    @staticmethod
    def split_temporal(
        df: pd.DataFrame,
        date_column: str = 'date',
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разделить данные по времени (для временных рядов)
        
        Args:
            df: DataFrame с данными
            date_column: Колонка с датой
            train_ratio: Доля train выборки
            val_ratio: Доля val выборки
            test_ratio: Доля test выборки
        
        Returns:
            (train, val, test) DataFrames
        """
        assert train_ratio + val_ratio + test_ratio == 1.0, \
            "Ratios must sum to 1.0"
        
        # Сортировать по дате
        df = df.sort_values(date_column)
        
        n = len(df)
        train_idx = int(n * train_ratio)
        val_idx = train_idx + int(n * val_ratio)
        
        train = df.iloc[:train_idx]
        val = df.iloc[train_idx:val_idx]
        test = df.iloc[val_idx:]
        
        logger.info(f"Split temporal: train={len(train)}, val={len(val)}, test={len(test)}")
        
        return train, val, test
    
    @staticmethod
    def split_random(
        df: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Случайное разделение данных
        
        Args:
            df: DataFrame с данными
            train_ratio: Доля train выборки
            val_ratio: Доля val выборки
            test_ratio: Доля test выборки
            seed: Seed для random state
        
        Returns:
            (train, val, test) DataFrames
        """
        assert train_ratio + val_ratio + test_ratio == 1.0, \
            "Ratios must sum to 1.0"
        
        np.random.seed(seed)
        
        n = len(df)
        indices = np.random.permutation(n)
        
        train_idx = int(n * train_ratio)
        val_idx = train_idx + int(n * val_ratio)
        
        train = df.iloc[indices[:train_idx]]
        val = df.iloc[indices[train_idx:val_idx]]
        test = df.iloc[indices[val_idx:]]
        
        logger.info(f"Split random: train={len(train)}, val={len(val)}, test={len(test)}")
        
        return train, val, test
    
    @staticmethod
    def stratified_split(
        df: pd.DataFrame,
        target_column: str = 'target',
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Стратифицированное разделение по таргету (для дисбалансированных данных)
        
        Args:
            df: DataFrame с данными
            target_column: Колонка с таргетом
            train_ratio: Доля train выборки
            val_ratio: Доля val выборки
            test_ratio: Доля test выборки
            seed: Seed для random state
        
        Returns:
            (train, val, test) DataFrames
        """
        from sklearn.model_selection import train_test_split
        
        # Первое разделение: train vs val+test
        train, temp = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=df[target_column],
            random_state=seed,
        )
        
        # Второе разделение: val vs test
        val, test = train_test_split(
            temp,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp[target_column],
            random_state=seed,
        )
        
        logger.info(f"Split stratified: train={len(train)}, val={len(val)}, test={len(test)}")
        logger.info(f"Class distribution in train: {train[target_column].value_counts().to_dict()}")
        logger.info(f"Class distribution in val: {val[target_column].value_counts().to_dict()}")
        logger.info(f"Class distribution in test: {test[target_column].value_counts().to_dict()}")
        
        return train, val, test


class SampleWeightComputer:
    """
    Вычисление весов для каждого примера
    """
    
    @staticmethod
    def temporal_weights(
        df: pd.DataFrame,
        date_column: str = 'date',
        decay_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Вычислить веса на основе возраста примера
        
        Более новые примеры получают больший вес
        
        Args:
            df: DataFrame с данными
            date_column: Колонка с датой
            decay_factor: Коэффициент экспоненциального спада
        
        Returns:
            Массив весов (длина = len(df))
        """
        dates = pd.to_datetime(df[date_column])
        max_date = dates.max()
        
        # Дни с максимальной даты
        days_diff = (max_date - dates).dt.days
        
        # Экспоненциальный спад + минимальный вес
        weights = np.exp(-days_diff / (365 * decay_factor)) + 0.5
        
        # Нормализовать
        weights = weights / weights.sum() * len(weights)
        
        return weights
    
    @staticmethod
    def class_weights(
        df: pd.DataFrame,
        target_column: str = 'target',
    ) -> np.ndarray:
        """
        Вычислить веса для балансировки классов
        
        Args:
            df: DataFrame с данными
            target_column: Колонка с таргетом
        
        Returns:
            Массив весов (длина = len(df))
        """
        from sklearn.utils.class_weight import compute_sample_weight
        
        weights = compute_sample_weight(
            'balanced',
            df[target_column].values,
        )
        
        return weights
    
    @staticmethod
    def combined_weights(
        df: pd.DataFrame,
        date_column: str = 'date',
        target_column: str = 'target',
        temporal_weight: float = 0.7,
        class_weight: float = 0.3,
    ) -> np.ndarray:
        """
        Комбинированные веса (временные + по классам)
        
        Args:
            df: DataFrame с данными
            date_column: Колонка с датой
            target_column: Колонка с таргетом
            temporal_weight: Вес временных весов
            class_weight: Вес весов классов
        
        Returns:
            Массив весов
        """
        tw = SampleWeightComputer.temporal_weights(df, date_column)
        cw = SampleWeightComputer.class_weights(df, target_column)
        
        # Нормализовать каждый вектор весов
        tw = tw / tw.mean()
        cw = cw / cw.mean()
        
        # Комбинировать
        weights = temporal_weight * tw + class_weight * cw
        
        # Финальная нормализация
        weights = weights / weights.mean() * len(weights)
        
        return weights
