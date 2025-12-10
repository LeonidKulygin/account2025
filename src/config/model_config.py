
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from enum import Enum


class RNNType(str, Enum):
    """Типы RNN"""
    LSTM = "LSTM"
    GRU = "GRU"
    RNN = "RNN"


class PoolingType(str, Enum):
    """Типы pooling операций"""
    MAX = "max"
    AVG = "avg"
    BOTH = "both"


@dataclass
class EmbeddingConfig:
    # Word2Vec описания
    description_embedding_dim: int = 50
    description_embedding_type: str = "word2vec"  # word2vec, tfidf, charcnn
    use_tfidf: bool = True
    tfidf_max_features: int = 40000
    
    # Entity Embeddings для категориальных признаков
    use_entity_embedding: bool = True
    embedding_formula: str = "sqrt"  # sqrt, log2, fixed
    
    # Размеры эмбеддингов для каждого признака
    embedding_sizes: Dict[str, int] = field(default_factory=dict)


@dataclass
class RNNConfig:
    """Конфиг для RNN слоя"""
    # Тип RNN
    rnn_type: RNNType = RNNType.GRU
    
    # Размеры
    input_size: int = 50
    hidden_size: int = 128
    num_layers: int = 2
    
    # Параметры RNN
    bidirectional: bool = True
    dropout: float = 0.3
    batch_first: bool = True
    
    # Выход из RNN
    pooling_type: PoolingType = PoolingType.BOTH


@dataclass
class DenseConfig:
    """Конфиг для плотных слоев"""
    # Размеры слоев
    dense_sizes: List[int] = field(default_factory=lambda: [512, 256, 64])
    dropout_rate: float = 0.3
    activation: str = "relu"  # relu, elu, gelu
    use_batch_norm: bool = False
    
    # Output
    output_size: int = 1


@dataclass
class ModelConfig:
    """Главная конфигурация модели"""
    
    # Компоненты
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rnn: RNNConfig = field(default_factory=RNNConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    
    # Общие параметры
    dropout_spatial: float = 0.2
    use_layer_norm: bool = False
    
    # Инициализация
    init_method: str = "xavier"  # xavier, kaiming, normal
    
    def to_dict(self) -> Dict:
        """Конвертировать в словарь"""
        return {
            'embedding': self.embedding.__dict__,
            'rnn': self.rnn.__dict__,
            'dense': self.dense.__dict__,
            'dropout_spatial': self.dropout_spatial,
            'use_layer_norm': self.use_layer_norm,
            'init_method': self.init_method,
        }
    
    @classmethod
    def get_baseline(cls):
        """Получить baseline конфиг"""
        return cls(
            embedding=EmbeddingConfig(use_tfidf=True),
            rnn=RNNConfig(
                rnn_type=RNNType.GRU,
                hidden_size=128,
                bidirectional=True,
                num_layers=2,
            ),
            dense=DenseConfig(dense_sizes=[512, 256, 64]),
        )
    
    @classmethod
    def get_lstm_advanced(cls):
        """Получить продвинутый LSTM конфиг"""
        return cls(
            embedding=EmbeddingConfig(use_tfidf=True),
            rnn=RNNConfig(
                rnn_type=RNNType.LSTM,
                hidden_size=256,
                bidirectional=True,
                num_layers=3,
                dropout=0.4,
            ),
            dense=DenseConfig(dense_sizes=[512, 256, 128, 64]),
        )
    
    @classmethod
    def get_lightweight(cls):
        """Получить легкий конфиг для быстрого обучения"""
        return cls(
            embedding=EmbeddingConfig(use_tfidf=False),
            rnn=RNNConfig(
                rnn_type=RNNType.GRU,
                hidden_size=64,
                bidirectional=False,
                num_layers=1,
            ),
            dense=DenseConfig(dense_sizes=[128, 64]),
        )
