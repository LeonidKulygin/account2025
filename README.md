# Credit Risk Prediction Model

Нейросетевая модель для прогнозирования кредитного риска клиентов банка на основе транзакционной истории.

Сделал Кулыгин Леонид 

## Структура проекта

```
account2025/
├── src/
│   ├── config/              # Конфигурации (модель, обучение, данные)
│   ├── data/                # Загрузка и обработка данных
│   ├── model/               # Архитектура модели (factory, modules, wrapper)
│   ├── training/            # Обучение (trainer, optimizers, schedulers)
│   ├── evaluation/          # Метрики и оценка
│   └── utils/               # Утилиты (loggers, ClearML, metrics)
├── train.py                 # Основной скрипт обучения
├── inference.py             # Скрипт для inference
├── requirements.txt         # Зависимости
└── README.md               # Этот файл
```

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Обучение модели с базовой конфигурацией

```bash
python train.py \
    --config src/config/experiments/baseline.yaml \
    --train-data data/train.csv \
    --val-data data/val.csv
```

### 3. Обучение с продвинутой конфигурацией (LSTM)

```bash
python train.py \
    --config src/config/experiments/lstm_advanced.yaml \
    --train-data data/train.pq \
    --val-data data/val.pq
```

### 4. Inference на тестовых данных

```bash
python inference.py \
    --model-path outputs/checkpoints/best.pth \
    --data-path data/test.csv \
    --output-path predictions.csv
```

##  Конфигурация

### Компоненты конфигурации

#### 1. Конфигурация модели (`model_config.py`)

```python
# RNN типы
- GRU (по умолчанию)
- LSTM
- RNN

# Параметры RNN
- hidden_size: 64-512 (обычно 128-256)
- num_layers: 1-4 (обычно 2-3)
- bidirectional: True/False
- dropout: 0.0-0.5

# Entity Embeddings
- use_entity_embedding: True/False
- embedding_formula: "sqrt", "log2", "fixed"

# Dense слои
- dense_sizes: [512, 256, 64] (настраивается)
- activation: "relu", "elu", "gelu"
- dropout_rate: 0.0-0.5
```

#### 2. Конфигурация обучения (`training_config.py`)

```python
# Оптимайзеры
- adam
- adamw (рекомендуется)
- sgd
- radam

# Scheduler'ы
- constant
- linear
- cosine (рекомендуется для качества)
- cyclical (рекомендуется для speed)
- exponential
- step

# Параметры обучения
- learning_rate: 1e-4 - 1e-2
- batch_size: 32-256
- num_epochs: 20-100
- weight_decay: 1e-6 - 1e-3
```

### Примеры предустановленных конфигов

**Baseline (быстро, хорошее качество):**
```yaml
# src/config/experiments/baseline.yaml
RNN: BiGRU
hidden_size: 128
layers: 2
optimizer: AdamW
scheduler: Cyclical
num_epochs: 50
```

**Advanced LSTM (лучшее качество, медленнее):**
```yaml
# src/config/experiments/lstm_advanced.yaml
RNN: BiLSTM
hidden_size: 256
layers: 3
optimizer: AdamW
scheduler: Cosine
num_epochs: 100
```

**Lightweight (быстро, для прототипирования):**
```python
from src.model.factory import PresetConfigs
model = PresetConfigs.get_lightweight()
```

## 🏗️ Архитектура модели

### Flow данных

```
Транзакции
    ↓
[Word2Vec + TF-IDF] → Эмбеддинги описаний (50 dim)
    ↓
[Entity Embeddings] → Категориальные признаки
    ↓
[BiGRU/BiLSTM] → RNN обработка (128-256 hidden)
    ↓
[Max/Avg Pooling] → История эмбеддинг
    ↓
Признаки предложения
    ↓
[Dense layers] → Классификатор (512→256→64→1)
    ↓
[Sigmoid] → Вероятность дефолта
```

### Ключевые компоненты

1. **EmbeddingLayer** - обработка текстовых описаний
2. **EntityEmbedding** - эмбеддинги для категориальных признаков
3. **RNNEncoder** - BiGRU/BiLSTM для последовательностей
4. **DenseClassifier** - финальная классификация

##  Метрики

### Поддерживаемые метрики

- **Gini** - метрика Джини (основная для банков)
- **ROC-AUC** - кривая под ROC
- **Precision / Recall** - точность и полнота
- **F1-Score** - гармоническое среднее
- **KS-Statistic** - статистика Колмогорова-Смирнова

### Логирование метрик с ClearML

```python
from src.utils.clearml_utils import ClearMLLogger

logger = ClearMLLogger(
    project_name="credit-risk",
    task_name="baseline-experiment"
)

# Логировать конфиг
logger.log_config(config_dict)

# Логировать метрики
logger.log_metrics({"train/loss": 0.45, "val/gini": 55.2}, step=epoch)

# Логировать модель
logger.log_model(model_path, "best_model")
```

##  Factory и конфигурирование

### Создание модели через Factory

```python
from src.config.model_config import ModelConfig
from src.model.factory import ModelFactory

# Создать конфиг
config = ModelConfig(
    rnn.hidden_size=256,
    rnn.num_layers=3,
    rnn.rnn_type="LSTM",
    rnn.bidirectional=True,
)

# Создать модель
model = ModelFactory.create(config)
```

### Использование предустановок

```python
from src.model.factory import PresetConfigs

# Базовая модель
model = PresetConfigs.get_baseline()

# LSTM advanced
model = PresetConfigs.get_lstm_advanced()

# Lightweight
model = PresetConfigs.get_lightweight()

# BiGRU с вниманием
model = PresetConfigs.get_bidgru_with_attention()
```

##  Оптимизаторы и Scheduler'ы

### Factory для оптимайзеров

```python
from src.training.optimizer_factory import create_optimizer
from src.config.training_config import OptimizerConfig

config = OptimizerConfig(
    optimizer_type="adamw",
    learning_rate=1e-3,
    weight_decay=1e-4,
)

optimizer = create_optimizer(model.parameters(), config)
```

### Factory для scheduler'ов

```python
from src.training.scheduler_factory import create_scheduler
from src.config.training_config import SchedulerConfig

config = SchedulerConfig(
    scheduler_type="cyclical",
    base_lr=1e-3,
    max_lr=1e-2,
    cycle_size=4,
)

scheduler = create_scheduler(optimizer, config, num_epochs=50)
```

### Cyclical Learning Rate

Рекомендуется для быстрого обучения и выхода из локальных минимумов:

```yaml
scheduler:
  scheduler_type: "cyclical"
  base_lr: 0.001       # Низкий LR
  max_lr: 0.01         # Высокий LR
  cycle_size: 4        # Цикл каждые 4 эпохи
```

### Cosine Annealing

Рекомендуется для лучшего финального качества:

```yaml
scheduler:
  scheduler_type: "cosine"
  t_max: 100           # Максимум эпох
  eta_min: 0.000001    # Минимальный LR
```

##  Экспериментирование

### Поиск гиперпараметров

```bash
python hyperparameter_search.py \
    --train-data data/train.csv \
    --val-data data/val.csv \
    --search-type grid  # или random
```

### Сравнение конфигов

```bash
# Запустить несколько конфигов в последовательности
python train.py --config src/config/experiments/baseline.yaml
python train.py --config src/config/experiments/lstm_advanced.yaml

# Результаты доступны в ClearML Dashboard
```

##  Примеры использования

### Пример 1: Обучение с базовой конфигурацией

```python
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.model.factory import ModelFactory
from train import TrainingPipeline

# Конфиги
model_config = ModelConfig.get_baseline()
training_config = TrainingConfig.get_baseline()

# Пайплайн
pipeline = TrainingPipeline(model_config, training_config)
pipeline.train(train_loader, val_loader)
```

### Пример 2: Кастомная конфигурация

```python
from src.config.model_config import (
    ModelConfig, RNNConfig, RNNType, DenseConfig
)
from src.config.training_config import (
    TrainingConfig, OptimizerConfig, SchedulerConfig,
    OptimizerType, SchedulerType
)

# Кастомная RNN конфигурация
rnn_config = RNNConfig(
    rnn_type=RNNType.LSTM,
    hidden_size=512,
    num_layers=4,
    bidirectional=True,
    dropout=0.4,
)

# Кастомная Dense конфигурация
dense_config = DenseConfig(
    dense_sizes=[1024, 512, 256, 128],
    dropout_rate=0.3,
    activation="gelu",
)

# Объединить в ModelConfig
model_config = ModelConfig(
    rnn=rnn_config,
    dense=dense_config,
    dropout_spatial=0.3,
)

# Кастомная training конфигурация
training_config = TrainingConfig(
    optimizer=OptimizerConfig(
        optimizer_type=OptimizerType.ADAMW,
        learning_rate=5e-4,
        weight_decay=1e-5,
    ),
    scheduler=SchedulerConfig(
        scheduler_type=SchedulerType.COSINE,
        t_max=200,
    ),
    num_epochs=150,
    batch_size=32,
)
```

### Пример 3: Inference

```python
from inference import CreditRiskInference
from src.config.model_config import ModelConfig

# Загрузить модель
model_config = ModelConfig.get_baseline()
inference = CreditRiskInference(
    model_path="outputs/checkpoints/best.pth",
    model_config=model_config,
    device="cuda"
)

# Сделать предсказание
predictions = inference.predict(
    transaction_sequences=batch_trans,
    categorical_features=batch_cat,
    offer_features=batch_offer,
)
```

##  Отладка и анализ

### Включить debug режим

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Просмотреть архитектуру модели

```python
from src.model.factory import ModelFactory
from src.config.model_config import ModelConfig

config = ModelConfig.get_baseline()
model = ModelFactory.create(config)
print(model)

# Подсчитать параметры
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
```

### Анализ метрик

```python
from src.utils.clearml_utils import MetricsLogger

metrics_logger = MetricsLogger()
history = metrics_logger.get_history()


metrics_logger.save()
```

## Best Practices

1. **Начните с baseline** - используйте `ModelConfig.get_baseline()` для первого прототипа
2. **Cyclical LR для speed** - `SchedulerType.CYCLICAL` для быстрого обучения
3. **Cosine для качества** - `SchedulerType.COSINE` для финального качества
4. **AdamW вместо Adam** - правильная L2 регуляризация
5. **Логируйте в ClearML** - отслеживайте все эксперименты
6. **Используйте early stopping** - избежите переобучения
7. **Сохраняйте checkpoints** - лучшую и последнюю модель
