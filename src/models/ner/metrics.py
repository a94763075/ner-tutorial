import numpy as np
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.base.metrics import BaseMetrics

class NERMetrics(BaseMetrics):
    """
    NER模型的評估指標計算類別，負責計算NER任務的各種評估指標。
    """
    def __init__(self, id2label):
        """
        初始化NER評估指標計算器。
        
        參數:
            id2label (dict): 將ID映射到標籤的字典。
        """
        super().__init__()
        self.id2label = id2label
        
    def compute_metrics(self, p):
        """
        計算NER模型的評估指標。
        
        參數:
            p (tuple): 包含預測結果和真實標籤的元組 (predictions, labels)。
            
        返回:
            dict: 包含各種評估指標的字典。
        """
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # 移除特殊標記的預測和標籤
        true_predictions = [
            [self.id2label[pred] for (pred, lbl) in zip(prediction, label) if lbl != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[lbl] for (pred, lbl) in zip(prediction, label) if lbl != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        return {
            "accuracy": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
