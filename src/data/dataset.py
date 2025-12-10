import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TransactionSequenceDataset(torch.utils.data.Dataset):
    """
    Dataset для работы с последовательностями транзакций
    
    Формат данных:
    - Каждая строка - одна заявка с историей транзакций
    - Колонки: client_id, date, description, amount, send_flag, target
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        word2vec_model=None,
        tfidf_vectorizer=None,
        max_seq_length: int = 550,
        use_tfidf: bool = True,
        categorical_columns: Optional[List[str]] = None,
        amount_buckets: Optional[np.ndarray] = None,
        days_before_buckets: Optional[np.ndarray] = None,
    ):
        """
        Инициализация Dataset
        
        Args:
            data: DataFrame с данными
            word2vec_model: Обученная word2vec модель
            tfidf_vectorizer: Fitted TF-IDF vectorizer
            max_seq_length: Максимальная длина последовательности
            use_tfidf: Использовать ли TF-IDF для взвешивания
            categorical_columns: Колонки для категориального encoding
            amount_buckets: Квантили для бакетизации суммы
            days_before_buckets: Квантили для бакетизации дней до скоринга
        """
        self.data = data
        self.word2vec_model = word2vec_model
        self.tfidf_vectorizer = tfidf_vectorizer
        self.max_seq_length = max_seq_length
        self.use_tfidf = use_tfidf
        self.categorical_columns = categorical_columns or []
        self.amount_buckets = amount_buckets
        self.days_before_buckets = days_before_buckets
        
        # Инициализировать category encoders
        self._setup_categorical_encoders()
        
        logger.info(f"Dataset initialized with {len(self.data)} samples")
    
    def _setup_categorical_encoders(self):
        """Создать категориальные encoders"""
        self.categorical_encoders = {}
        
        for col in self.categorical_columns:
            if col in self.data.columns:
                unique_values = self.data[col].unique()
                encoder = {val: idx for idx, val in enumerate(unique_values)}
                self.categorical_encoders[col] = encoder
                logger.info(f"Created encoder for {col} with {len(encoder)} unique values")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Получить один элемент датасета
        
        Returns:
            Dict с ключами:
            - transactions: (seq_len, feature_dim) - эмбеддинги транзакций
            - categorical_features: (num_categorical,) - категориальные признаки
            - offer_features: (num_offer_features,) - признаки предложения
            - lengths: int - длина последовательности
            - targets: float - таргет (дефолт/нет)
        """
        row = self.data.iloc[idx]
        
        # Получить историю транзакций
        transactions = self._get_transaction_embeddings(row)
        
        # Получить категориальные признаки
        categorical_features = self._get_categorical_features(row)
        
        # Получить признаки предложения
        offer_features = self._get_offer_features(row)
        
        # Получить таргет
        target = float(row.get('target', 0))
        
        # Длина последовательности
        seq_length = len(transactions)
        
        return {
            'transactions': torch.FloatTensor(transactions),
            'categorical_features': torch.LongTensor(categorical_features),
            'offer_features': torch.LongTensor(offer_features),
            'lengths': torch.LongTensor([seq_length]),
            'targets': torch.FloatTensor([target]),
        }
    
    def _get_transaction_embeddings(self, row) -> np.ndarray:
        """
        Получить эмбеддинги для последовательности транзакций
        
        Returns:
            (seq_len, embedding_dim) - эмбеддинги
        """
        # TODO: Реализовать в зависимости от формата данных
        # Это может быть list of dicts или list of arrays
        
        # Пример с простыми эмбеддингами
        if 'transactions' not in row:
            # Если нет отдельной колонки transactions, создать пустой
            return np.zeros((1, 50), dtype=np.float32)
        
        transactions = row['transactions']
        
        # Ограничить длину
        if len(transactions) > self.max_seq_length:
            transactions = transactions[-self.max_seq_length:]
        
        embeddings = []
        
        for trans in transactions:
            if isinstance(trans, dict):
                # Обработать текстовое описание
                emb = self._embed_transaction(trans)
            else:
                # Просто скопировать эмбеддинг
                emb = trans
            
            embeddings.append(emb)
        
        if not embeddings:
            # Если пусто, вернуть нулевой эмбеддинг
            return np.zeros((1, 50), dtype=np.float32)
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Убедиться что правильная размерность
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)
        
        return embeddings
    
    def _embed_transaction(self, transaction: Dict) -> np.ndarray:
        """
        Создать эмбеддинг для одной транзакции
        
        Args:
            transaction: Dict с полями description, amount, send_flag, и т.д.
        
        Returns:
            Эмбеддинг размера embedding_dim
        """
        # Получить текстовое описание
        description = transaction.get('description', '')
        
        if not description or self.word2vec_model is None:
            # Вернуть нулевой эмбеддинг
            return np.zeros(50, dtype=np.float32)
        
        # Токенизировать и получить word2vec эмбеддинги
        tokens = description.split()
        
        if not tokens:
            return np.zeros(50, dtype=np.float32)
        
        # Получить word vectors
        vectors = []
        weights = []
        
        for token in tokens:
            if token in self.word2vec_model.wv:
                vectors.append(self.word2vec_model.wv[token])
                weights.append(1.0)
        
        if not vectors:
            return np.zeros(50, dtype=np.float32)
        
        vectors = np.array(vectors)
        weights = np.array(weights)
        
        # Применить TF-IDF веса если нужно
        if self.use_tfidf and self.tfidf_vectorizer is not None:
            # TODO: Получить TF-IDF веса
            pass
        
        # Взвешенное усреднение
        weights = weights / weights.sum()
        embedding = (vectors * weights[:, np.newaxis]).sum(axis=0)
        
        return embedding.astype(np.float32)
    
    def _get_categorical_features(self, row) -> np.ndarray:
        """
        Получить категориальные признаки
        
        Returns:
            (num_categorical,) - закодированные категориальные признаки
        """
        features = []
        
        for col in self.categorical_columns:
            if col in row:
                value = row[col]
                
                if col in self.categorical_encoders:
                    # Закодировать
                    encoded = self.categorical_encoders[col].get(value, 0)
                else:
                    encoded = 0
                
                features.append(encoded)
        
        return np.array(features, dtype=np.int64)
    
    def _get_offer_features(self, row) -> np.ndarray:
        """
        Получить признаки предложения (не зависят от истории)
        
        Returns:
            (num_offer_features,) - признаки предложения
        """
        features = []
        
        # Пример: месяц заявки
        if 'date' in row:
            try:
                date = pd.to_datetime(row['date'])
                month = date.month  # 1-12
                features.append(month - 1)  # 0-11
            except:
                features.append(0)
        
        # Пример: длина истории
        if 'transactions' in row:
            seq_len = min(len(row['transactions']), self.max_seq_length)
            # Бакетизировать длину
            if self.days_before_buckets is not None:
                seq_len_bucket = np.searchsorted(self.days_before_buckets, seq_len)
            else:
                # Простое бакетирование: 0-50, 50-100, 100-200, 200+
                seq_len_bucket = min(seq_len // 50, 10)
            features.append(seq_len_bucket)
        else:
            features.append(0)
        
        return np.array(features, dtype=np.int64)


class TransactionBatchDataset(torch.utils.data.Dataset):
    """
    Альтернативный Dataset для уже предобработанных данных
    
    Ожидает данные в виде numpy arrays или tensors
    """
    
    def __init__(
        self,
        transactions: np.ndarray,
        categorical_features: np.ndarray,
        offer_features: np.ndarray,
        targets: np.ndarray,
        lengths: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ):
        """
        Инициализация Dataset
        
        Args:
            transactions: (N, max_seq_len, feature_dim) - эмбеддинги транзакций
            categorical_features: (N, num_cat_features) - категориальные признаки
            offer_features: (N, num_offer_features) - признаки предложения
            targets: (N,) - таргеты
            lengths: (N,) - длины последовательностей
            weights: (N,) - веса для каждого примера
        """
        self.transactions = transactions
        self.categorical_features = categorical_features
        self.offer_features = offer_features
        self.targets = targets
        self.lengths = lengths
        self.weights = weights
        
        # Конвертировать в float32 если нужно
        if not isinstance(transactions, torch.Tensor):
            self.transactions = torch.FloatTensor(transactions)
        if not isinstance(categorical_features, torch.Tensor):
            self.categorical_features = torch.LongTensor(categorical_features)
        if not isinstance(offer_features, torch.Tensor):
            self.offer_features = torch.LongTensor(offer_features)
        if not isinstance(targets, torch.Tensor):
            self.targets = torch.FloatTensor(targets)
        
        if lengths is not None and not isinstance(lengths, torch.Tensor):
            self.lengths = torch.LongTensor(lengths)
        
        if weights is not None and not isinstance(weights, torch.Tensor):
            self.weights = torch.FloatTensor(weights)
        
        assert len(self.transactions) == len(self.targets), \
            f"Mismatch: {len(self.transactions)} != {len(self.targets)}"
        
        logger.info(f"BatchDataset initialized with {len(self)} samples")
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Dict:
        return {
            'transactions': self.transactions[idx],
            'categorical_features': self.categorical_features[idx],
            'offer_features': self.offer_features[idx],
            'targets': self.targets[idx],
            'lengths': self.lengths[idx] if self.lengths is not None else torch.LongTensor([1]),
            'weights': self.weights[idx] if self.weights is not None else torch.FloatTensor([1.0]),
        }


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Custom Sampler для группирования последовательностей по длине
    
    Это уменьшает количество padding и ускоряет обучение
    """
    
    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        num_buckets: int = 10,
        drop_last: bool = False,
    ):
        """
        Инициализация sampler
        
        Args:
            lengths: (N,) - длины последовательностей
            batch_size: Размер батча
            num_buckets: Количество бакетов для группировки
            drop_last: Выбросить ли последний неполный батч
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Разделить на бакеты по длине
        self.buckets = self._create_buckets(num_buckets)
    
    def _create_buckets(self, num_buckets: int) -> List[List[int]]:
        """
        Создать бакеты по длине последовательностей
        
        Returns:
            List of lists - индексы сгруппированные по бакетам
        """
        # Найти границы бакетов
        bucket_boundaries = np.percentile(
            self.lengths,
            np.linspace(0, 100, num_buckets + 1)
        )
        
        buckets = [[] for _ in range(num_buckets)]
        
        for idx, length in enumerate(self.lengths):
            bucket_idx = np.searchsorted(bucket_boundaries, length) - 1
            bucket_idx = min(bucket_idx, num_buckets - 1)
            buckets[bucket_idx].append(idx)
        
        return buckets
    
    def __iter__(self):
        """Итератор по батчам"""
        for bucket in self.buckets:
            # Перемешать индексы в батче
            bucket = list(bucket)
            np.random.shuffle(bucket)
            
            # Создать батчи из этого бакета
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch
    
    def __len__(self) -> int:
        total = sum(
            (len(bucket) + self.batch_size - 1) // self.batch_size
            for bucket in self.buckets
        )
        return total
