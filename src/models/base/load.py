from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseLoader(ABC):
    """
    模型載入的基類，為所有模型類型提供通用介面。
    """
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> Any:
        """
        抽象方法，用於載入模型。必須由子類實現。
        
        參數:
            model_path (str): 模型檔案或目錄的路徑。
            **kwargs: 其他載入參數。
            
        返回:
            載入的模型物件。
        """
        pass

    @abstractmethod
    def load_tokenizer(self, tokenizer_path: str, **kwargs) -> Any:
        """
        抽象方法，用於載入分詞器。必須由子類實現。
        
        參數:
            tokenizer_path (str): 分詞器檔案或目錄的路徑。
            **kwargs: 其他載入參數。
            
        返回:
            載入的分詞器物件。
        """
        pass
