from typing import Any, Optional
from transformers import AutoModelForTokenClassification, AutoTokenizer
from src.models.base.load import BaseLoader

class NERLoader(BaseLoader):
    """
    NER模型的載入器，繼承自BaseLoader，實現NER特定的模型和分詞器載入邏輯。
    """
    def load_model(self, model_path: str, **kwargs) -> Any:
        """
        載入NER模型。
        
        參數:
            model_path (str): 模型檔案或目錄的路徑。
            **kwargs: 其他載入參數。
            
        返回:
            載入的NER模型物件。
        """
        model = AutoModelForTokenClassification.from_pretrained(model_path, **kwargs)
        print(f"NER模型已從 {model_path} 載入。")
        return model

    def load_tokenizer(self, tokenizer_path: str, **kwargs) -> Any:
        """
        載入NER分詞器。
        
        參數:
            tokenizer_path (str): 分詞器檔案或目錄的路徑。
            **kwargs: 其他載入參數。
            
        返回:
            載入的NER分詞器物件。
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
        print(f"NER分詞器已從 {tokenizer_path} 載入。")
        return tokenizer
