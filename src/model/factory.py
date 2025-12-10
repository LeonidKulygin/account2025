import torch
import torch.nn as nn
from typing import Dict, Optional, Type
from .modules import (
    EmbeddingLayer, 
    RNNEncoder, 
    DenseClassifier,
    EntityEmbedding
)
from .wrapper import CreditRiskModel
from ..config.model_config import ModelConfig, RNNType


class ModelFactory:
    """Factory для создания моделей"""
    
    # Маппинг типов RNN
    RNN_TYPES = {
        RNNType.LSTM: nn.LSTM,
        RNNType.GRU: nn.GRU,
        RNNType.RNN: nn.RNN,
    }
    
    def __init__(self, config: ModelConfig):
        """Инициализация factory"""
        self.config = config
    
    @staticmethod
    def create(config: ModelConfig) -> CreditRiskModel:
        """Создать модель из конфига"""
        factory = ModelFactory(config)
        return factory.build()
    
    def build(self) -> CreditRiskModel:
        """Построить полную модель"""
        
        # 1. Embedding слой для описаний
        embedding_layer = EmbeddingLayer(
            embedding_dim=self.config.embedding.description_embedding_dim,
            embedding_type=self.config.embedding.description_embedding_type,
            use_tfidf=self.config.embedding.use_tfidf,
        )
        
        # 2. Entity Embeddings для категориальных признаков
        entity_embeddings = EntityEmbedding(
            feature_dims=self.config.embedding.embedding_sizes,
            embedding_formula=self.config.embedding.embedding_formula,
        ) if self.config.embedding.use_entity_embedding else None
        
        # 3. RNN encoder
        rnn_encoder = RNNEncoder(
            rnn_type=self.config.rnn.rnn_type,
            input_size=self._calculate_input_size(),
            hidden_size=self.config.rnn.hidden_size,
            num_layers=self.config.rnn.num_layers,
            bidirectional=self.config.rnn.bidirectional,
            dropout=self.config.rnn.dropout,
            pooling_type=self.config.rnn.pooling_type,
        )
        
        # 4. Dense классификатор
        dense_classifier = DenseClassifier(
            input_size=self._calculate_dense_input_size(),
            dense_sizes=self.config.dense.dense_sizes,
            dropout_rate=self.config.dense.dropout_rate,
            activation=self.config.dense.activation,
            use_batch_norm=self.config.dense.use_batch_norm,
            output_size=self.config.dense.output_size,
        )
        
        # 5. Обернуть в основной класс модели
        model = CreditRiskModel(
            embedding_layer=embedding_layer,
            rnn_encoder=rnn_encoder,
            dense_classifier=dense_classifier,
            entity_embeddings=entity_embeddings,
            dropout_spatial=self.config.dropout_spatial,
        )
        
        # 6. Инициализировать веса
        self._initialize_weights(model)
        
        return model
    
    def _calculate_input_size(self) -> int:
        """Расчитать размер входа для RNN"""
        # Описание (word2vec)
        input_size = self.config.embedding.description_embedding_dim
        
        # Эмбеддинги категориальных признаков
        if self.config.embedding.use_entity_embedding:
            # Сумма размеров всех эмбеддингов
            input_size += sum(self.config.embedding.embedding_sizes.values())
        
        return input_size
    
    def _calculate_dense_input_size(self) -> int:
        """Расчитать размер входа для Dense слоев"""
        # RNN выход
        rnn_output_size = self.config.rnn.hidden_size
        if self.config.rnn.bidirectional:
            rnn_output_size *= 2
        
        # Pooling
        if self.config.rnn.pooling_type == "both":
            rnn_output_size *= 2  # max и avg
        
        # Добавляем эмбеддинги признаков предложения
        offer_embedding_size = 16  # Можно сделать конфигурируемым
        
        return rnn_output_size + offer_embedding_size
    
    def _initialize_weights(self, model: nn.Module):
        """Инициализировать веса модели"""
        init_method = self.config.init_method
        
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_method == "normal":
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        if init_method == "orthogonal":
                            nn.init.orthogonal_(param)
                        else:
                            nn.init.uniform_(param, -0.1, 0.1)
            
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)


class PresetConfigs:
    """Предустановленные конфигурации моделей"""
    
    @staticmethod
    def get_baseline() -> CreditRiskModel:
        """Базовая модель"""
        config = ModelConfig.get_baseline()
        return ModelFactory.create(config)
    
    @staticmethod
    def get_lstm_advanced() -> CreditRiskModel:
        """Продвинутая LSTM модель"""
        config = ModelConfig.get_lstm_advanced()
        return ModelFactory.create(config)
    
    @staticmethod
    def get_lightweight() -> CreditRiskModel:
        """Легкая модель для быстрого прототипирования"""
        config = ModelConfig.get_lightweight()
        return ModelFactory.create(config)
    
    @staticmethod
    def get_bidgru_with_attention() -> CreditRiskModel:
        """BiGRU с механизмом внимания"""
        config = ModelConfig.get_baseline()
        config.rnn.bidirectional = True
        config.rnn.rnn_type = RNNType.GRU
        config.rnn.hidden_size = 256
        config.rnn.num_layers = 3
        return ModelFactory.create(config)
