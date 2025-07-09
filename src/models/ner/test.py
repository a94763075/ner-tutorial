from typing import Dict, Any, Optional, List
import torch
from transformers import Trainer
from src.models.base.test import BaseTester
from src.models.ner.data import NERDataProcessor

class NERTester(BaseTester):
    """
    NER模型的測試器，繼承自BaseTester，實現NER特定的評估和預測邏輯。
    """
    def __init__(self):
        """
        初始化NER測試器。
        """
        self.data_processor = NERDataProcessor()

    def evaluate(self, test_data_path: str, model: Any, **kwargs) -> Dict[str, float]:
        """
        評估NER模型效能。
        
        參數:
            test_data_path (str): 測試數據的路徑。
            model (Any): 已載入的模型物件。
            **kwargs: 其他評估參數，包含 'tokenizer' 和 'label2id'。
            
        返回:
            Dict[str, float]: 評估指標的字典，例如 {'loss': 0.5}。
        """
        tokenizer = kwargs.get('tokenizer')
        label2id = kwargs.get('label2id', {})
        
        if not tokenizer or not label2id:
            raise ValueError("評估NER模型需要提供 tokenizer 和 label2id 參數。")
        
        # 載入測試數據
        test_data = self.data_processor.load_data(test_data_path)
        test_dataset = self.data_processor.prepare_dataset(test_data, tokenizer=tokenizer, label2id=label2id)
        
        # 初始化訓練器用於評估
        trainer = Trainer(model=model, tokenizer=tokenizer)
        
        # 評估模型
        results = trainer.evaluate(test_dataset)
        print(f"評估結果: {results}")
        return results

    def predict(self, input_data: Any, model: Any, **kwargs) -> Any:
        """
        對輸入數據進行NER預測。
        
        參數:
            input_data (Any): 輸入數據，可以是文本或列表。
            model (Any): 已載入的模型物件。
            **kwargs: 其他預測參數，包含 'tokenizer' 和 'id2label'。
            
        返回:
            預測結果，格式為列表，包含識別的實體標籤。
        """
        tokenizer = kwargs.get('tokenizer')
        id2label = kwargs.get('id2label', {})
        
        if not tokenizer:
            raise ValueError("預測NER結果需要提供 tokenizer 參數。")
        
        # 處理輸入數據
        if isinstance(input_data, str):
            tokens = tokenizer(input_data, return_tensors="pt")
        elif isinstance(input_data, list):
            tokens = tokenizer(input_data, is_split_into_words=True, return_tensors="pt")
        else:
            raise ValueError("輸入數據必須是字符串或列表。")
        
        # 進行預測
        model.eval()
        with torch.no_grad():
            outputs = model(**tokens)
            predictions = outputs.logits.argmax(dim=-1)
        
        # 將預測結果轉換為標籤
        predicted_labels = []
        for pred in predictions[0].tolist():
            if pred in id2label:
                predicted_labels.append(id2label[pred])
            else:
                predicted_labels.append("O")
        
        print(f"預測結果: {predicted_labels}")
        return predicted_labels
