import torch
import torch.nn as nn
from typing import Optional, Tuple


class CreditRiskModel(nn.Module):
    def __init__(
        self,
        embedding_layer: nn.Module,
        rnn_encoder: nn.Module,
        dense_classifier: nn.Module,
        entity_embeddings: Optional[nn.Module] = None,
        dropout_spatial: float = 0.2,
    ):
        """
        Инициализация модели
        
        Args:
            embedding_layer: Слой для обработки описаний
            rnn_encoder: RNN энкодер для последовательностей
            dense_classifier: Dense классификатор
            entity_embeddings: Entity embeddings для категориальных признаков
            dropout_spatial: Spatial dropout для эмбеддингов
        """
        super().__init__()
        
        self.embedding_layer = embedding_layer
        self.rnn_encoder = rnn_encoder
        self.dense_classifier = dense_classifier
        self.entity_embeddings = entity_embeddings
        
        # Spatial Dropout
        self.spatial_dropout = nn.Dropout(dropout_spatial)
    
    def forward(
        self,
        transaction_descriptions: torch.Tensor,
        categorical_features: torch.Tensor,
        offer_features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            transaction_descriptions: Описания транзакций (batch, seq_len, max_len)
            categorical_features: Категориальные признаки (batch, seq_len, num_features)
            offer_features: Признаки предложения (batch, num_offer_features)
            lengths: Длины последовательностей (batch,)
        
        Returns:
            Предсказания (batch, 1)
        """
        batch_size, seq_len = transaction_descriptions.shape[0], transaction_descriptions.shape[1]
        
        # 1. Получить эмбеддинги описаний
        # (batch, seq_len, embedding_dim)
        desc_embeddings = self.embedding_layer(transaction_descriptions)
        
        # 2. Получить entity embeddings для категориальных признаков
        if self.entity_embeddings is not None:
            # (batch, seq_len, num_features) → (batch, seq_len, embedding_dim)
            cat_embeddings = self.entity_embeddings(categorical_features)
        else:
            cat_embeddings = None
        
        # 3. Объединить эмбеддинги
        if cat_embeddings is not None:
            # (batch, seq_len, embedding_dim + cat_embedding_dim)
            combined_embeddings = torch.cat([desc_embeddings, cat_embeddings], dim=-1)
        else:
            combined_embeddings = desc_embeddings
        
        # 4. Применить spatial dropout
        combined_embeddings = self.spatial_dropout(combined_embeddings)
        
        # 5. Пропустить через RNN
        # (batch, seq_len, rnn_hidden_size * 2) для BiRNN с pooling
        rnn_output = self.rnn_encoder(combined_embeddings, lengths)
        
        # 6. Процессить признаки предложения
        # offer_embeddings: (batch, offer_embedding_dim)
        offer_embeddings = self._process_offer_features(offer_features)
        
        # 7. Объединить историю и признаки предложения
        # (batch, rnn_output_dim + offer_embedding_dim)
        combined_features = torch.cat([rnn_output, offer_embeddings], dim=-1)
        
        # 8. Пропустить через классификатор
        # (batch, 1)
        predictions = self.dense_classifier(combined_features)
        
        return predictions
    
    def _process_offer_features(self, offer_features: torch.Tensor) -> torch.Tensor:
        """
        Обработать признаки предложения
        (может быть переопределено для более сложной обработки)
        
        Args:
            offer_features: (batch, num_offer_features)
        
        Returns:
            offer_embeddings: (batch, offer_embedding_dim)
        """
        # Простой linear слой
        return offer_features
    
    def get_embeddings(
        self,
        transaction_descriptions: torch.Tensor,
        categorical_features: torch.Tensor,
        offer_features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Получить эмбеддинги истории (до классификатора)
        Полезно для анализа и transfer learning
        
        Args:
            Те же что и в forward
        
        Returns:
            Эмбеддинги истории (batch, embedding_dim)
        """
        # Повторить логику forward до классификатора
        desc_embeddings = self.embedding_layer(transaction_descriptions)
        
        if self.entity_embeddings is not None:
            cat_embeddings = self.entity_embeddings(categorical_features)
        else:
            cat_embeddings = None
        
        if cat_embeddings is not None:
            combined_embeddings = torch.cat([desc_embeddings, cat_embeddings], dim=-1)
        else:
            combined_embeddings = desc_embeddings
        
        combined_embeddings = self.spatial_dropout(combined_embeddings)
        rnn_output = self.rnn_encoder(combined_embeddings, lengths)
        
        return rnn_output
