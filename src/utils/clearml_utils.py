from clearml import Task
from typing import Dict, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ClearMLLogger:
    """Логгер для ClearML"""
    
    def __init__(
        self,
        project_name: str = "credit-risk",
        task_name: str = "experiment",
        auto_connect_frameworks: bool = True,
    ):
        """Инициализация ClearML логгера"""
        self.project_name = project_name
        self.task_name = task_name
        
        try:
            # Инициализировать задачу
            self.task = Task.init(
                project_name=project_name,
                task_name=task_name,
                auto_connect_frameworks=auto_connect_frameworks,
            )
            self.logger = self.task.get_logger()
            logger.info(f"ClearML task initialized: {task_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize ClearML: {e}")
            self.task = None
            self.logger = None
    
    def log_config(self, config: Dict[str, Any]):
        """Логировать конфигурацию"""
        if self.task is not None:
            try:
                self.task.connect_configuration(config)
                logger.info("Config logged to ClearML")
            except Exception as e:
                logger.warning(f"Failed to log config: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Логировать метрики"""
        if self.logger is not None:
            try:
                for key, value in metrics.items():
                    # Парсим ключ типа "train/loss" -> ("train", "loss")
                    if "/" in key:
                        title, series = key.split("/", 1)
                    else:
                        title = "metrics"
                        series = key
                    
                    self.logger.report_scalar(
                        title=title,
                        series=series,
                        value=value,
                        iteration=step,
                    )
            except Exception as e:
                logger.warning(f"Failed to log metrics: {e}")
    
    def log_hyper_params(self, params: Dict[str, Any]):
        """Логировать гиперпараметры"""
        if self.task is not None:
            try:
                self.task.connect_parameters(params)
                logger.info("Hyperparameters logged to ClearML")
            except Exception as e:
                logger.warning(f"Failed to log hyperparameters: {e}")
    
    def log_artifact(self, artifact_path: Path, artifact_name: str = None):
        """Логировать артефакт (модель, данные, и т.д.)"""
        if self.task is not None:
            try:
                artifact_name = artifact_name or artifact_path.name
                self.task.upload_artifact(
                    name=artifact_name,
                    artifact_object=str(artifact_path),
                )
                logger.info(f"Artifact uploaded: {artifact_name}")
            except Exception as e:
                logger.warning(f"Failed to upload artifact: {e}")
    
    def log_model(self, model_path: Path, model_name: str = "model"):
        """Логировать модель"""
        if self.task is not None:
            try:
                self.task.upload_artifact(
                    name=model_name,
                    artifact_object=str(model_path),
                )
                logger.info(f"Model uploaded: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to upload model: {e}")
    
    def log_text(self, title: str, text: str):
        """Логировать текст (например, результаты анализа)"""
        if self.logger is not None:
            try:
                self.logger.report_text(title=title, series=title, text=text)
            except Exception as e:
                logger.warning(f"Failed to log text: {e}")
    
    def close_task(self):
        """Закончить задачу"""
        if self.task is not None:
            try:
                self.task.close()
                logger.info("ClearML task closed")
            except Exception as e:
                logger.warning(f"Failed to close task: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_task()


class MetricsLogger:
    """Логгер для метрик (не зависит от ClearML)"""
    
    def __init__(self, log_dir: Path = None):
        """Инициализация логгера метрик"""
        self.log_dir = log_dir or Path("outputs/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = {}
    
    def log_metrics(self, metrics: Dict[str, float], stage: str = "train", epoch: int = None):
        """Логировать метрики"""
        if stage not in self.metrics_history:
            self.metrics_history[stage] = {}
        
        if epoch is None:
            epoch = len(self.metrics_history[stage])
        
        for key, value in metrics.items():
            if key not in self.metrics_history[stage]:
                self.metrics_history[stage][key] = []
            self.metrics_history[stage][key].append(value)
    
    def get_history(self, stage: str = None) -> Dict:
        """Получить историю метрик"""
        if stage is None:
            return self.metrics_history
        return self.metrics_history.get(stage, {})
    
    def save(self, path: Path = None):
        """Сохранить историю в CSV"""
        import pandas as pd
        
        path = path or self.log_dir / "metrics_history.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Объединить все метрики в один DataFrame
        data = {}
        for stage, metrics in self.metrics_history.items():
            for key, values in metrics.items():
                full_key = f"{stage}/{key}"
                data[full_key] = values
        
        df = pd.DataFrame(data)
        df.to_csv(path, index_label="epoch")
        logger.info(f"Metrics saved to {path}")
