from abc import ABC, abstractmethod

class BaseMetrics(ABC):
    """
    基礎評估指標類別，定義評估指標計算的通用接口。
    """
    def __init__(self):
        """
        初始化基礎評估指標計算器。
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, p):
        """
        計算評估指標的抽象方法，子類必須實現。
        
        參數:
            p (tuple): 包含預測結果和真實標籤的元組。
            
        返回:
            dict: 包含各種評估指標的字典。
        """
        pass
