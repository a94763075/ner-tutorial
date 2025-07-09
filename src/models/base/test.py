from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseTester(ABC):
    """
    模型測試的基類，為所有模型類型提供通用介面。
    """
    @abstractmethod
    def evaluate(self, test_data_path: str, model: Any, **kwargs) -> Dict[str, float]:
        """
        抽象方法，用於評估模型效能。必須由子類實現。
        
        參數:
            test_data_path (str): 測試數據的路徑。
            model (Any): 已載入的模型物件。
            **kwargs: 其他評估參數。
            
        返回:
            Dict[str, float]: 評估指標的字典，例如 {'accuracy': 0.85, 'f1': 0.82}。
        """
        pass

    @abstractmethod
    def predict(self, input_data: Any, model: Any, **kwargs) -> Any:
        """
        抽象方法，用於對輸入數據進行預測。必須由子類實現。
        
        參數:
            input_data (Any): 輸入數據，可以是文本、列表或其他格式。
            model (Any): 已載入的模型物件。
            **kwargs: 其他預測參數。
            
        返回:
            預測結果，格式由子類定義。
        """
        pass
