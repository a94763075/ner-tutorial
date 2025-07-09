from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseDataProcessor(ABC):
    """
    數據處理的基類，為所有模型類型提供通用數據處理介面。
    """
    @abstractmethod
    def load_data(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        抽象方法，用於從檔案載入數據。必須由子類實現。
        
        參數:
            file_path (str): 數據檔案的路徑。
            **kwargs: 其他載入參數。
            
        返回:
            List[Dict[str, Any]]: 包含數據的列表，每個元素是一個字典。
        """
        pass

    @abstractmethod
    def prepare_dataset(self, data: List[Dict[str, Any]], **kwargs) -> Any:
        """
        抽象方法，用於將原始數據轉換為適合模型訓練或測試的格式。必須由子類實現。
        
        參數:
            data (List[Dict[str, Any]]): 原始數據列表。
            **kwargs: 其他處理參數，例如 tokenizer, label2id 等。
            
        返回:
            Any: 處理後的數據集，格式由子類定義。
        """
        pass

    @abstractmethod
    def get_unique_labels(self, data: List[Dict[str, Any]], **kwargs) -> List[str]:
        """
        抽象方法，用於從數據集中提取唯一的標籤。必須由子類實現。
        
        參數:
            data (List[Dict[str, Any]]): 數據列表。
            **kwargs: 其他參數。
            
        返回:
            List[str]: 排序後的唯一標籤列表。
        """
        pass
