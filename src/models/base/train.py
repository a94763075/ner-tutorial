import mlflow
import mlflow.pytorch
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseTrainer(ABC):
    """
    模型訓練的基類，為所有模型類型提供通用介面。
    整合 MLflow 用於實驗追蹤。
    """
    def __init__(self, experiment_name: str = "Default_Experiment", run_name: Optional[str] = None):
        """
        初始化訓練器，設置 MLflow 實驗。
        
        參數:
            experiment_name (str): MLflow 實驗的名稱。
            run_name (str, 可選): 當前運行的名稱。如果未提供，MLflow 將自動生成一個。
        """
        # 設置 MLflow 實驗
        mlflow.set_experiment(experiment_name)
        self.run_name = run_name
        self.active_run = None

    def start_run(self) -> None:
        """開始一個 MLflow 運行，用於追蹤訓練過程。"""
        if self.active_run is None:
            self.active_run = mlflow.start_run(run_name=self.run_name)
            print(f"MLflow 運行已開始，ID: {self.active_run.info.run_id}")

    def end_run(self) -> None:
        """結束當前的 MLflow 運行。"""
        if self.active_run is not None:
            mlflow.end_run()
            self.active_run = None
            print("MLflow 運行已結束。")

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        將訓練參數記錄到 MLflow。
        
        參數:
            params (dict): 要記錄的參數字典。
        """
        if self.active_run:
            mlflow.log_params(params)
            print("訓練參數已記錄到 MLflow。")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        將指標記錄到 MLflow。
        
        參數:
            key (str): 指標的名稱。
            value (float): 指標的值。
            step (int, 可選): 指標的步驟號。
        """
        if self.active_run:
            mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """
        將工件（例如模型檔案）記錄到 MLflow。
        
        參數:
            local_path (str): 工件的本地路徑。
            artifact_path (str, 可選): 工件儲存中的目標路徑。
        """
        if self.active_run:
            mlflow.log_artifact(local_path, artifact_path)
            print(f"工件 {local_path} 已記錄到 MLflow。")

    @abstractmethod
    def train(self, train_data_path: str, output_dir: str, **kwargs) -> None:
        """
        抽象方法，用於訓練模型。必須由子類實現。
        
        參數:
            train_data_path (str): 訓練數據的路徑。
            output_dir (str): 儲存訓練模型的目錄。
            **kwargs: 其他訓練參數。
        """
        pass

    @abstractmethod
    def save_model(self, output_dir: str) -> None:
        """
        抽象方法，用於儲存訓練好的模型。必須由子類實現。
        
        參數:
            output_dir (str): 儲存模型的目錄。
        """
        pass
