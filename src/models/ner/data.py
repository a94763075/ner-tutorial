import json
from typing import List, Dict, Any, Optional
from datasets import Dataset
from src.models.base.data import BaseDataProcessor

class NERDataProcessor(BaseDataProcessor):
    """
    NER模型的數據處理器，繼承自BaseDataProcessor，實現NER特定的數據處理邏輯。
    """
    def load_data(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        """
        從JSON檔案載入NER數據，每行是一個包含 'words' 和 'ner' 列表的JSON物件。
        
        參數:
            file_path (str): 數據檔案的路徑。
            **kwargs: 其他載入參數。
            
        返回:
            List[Dict[str, Any]]: 包含數據的列表，每個元素是一個字典，包含 'words' 和 'ner' 鍵。
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get('words') and entry.get('ner'):
                        data.append(entry)
        print(f"NER數據已從 {file_path} 載入，共 {len(data)} 條記錄。")
        return data

    def prepare_dataset(self, data: List[Dict[str, Any]], **kwargs) -> Dataset:
        """
        將原始NER數據轉換為適合使用transformers進行訓練或測試的格式。
        
        參數:
            data (List[Dict[str, Any]]): 原始數據列表。
            **kwargs: 其他處理參數，必須包含 'tokenizer' 和 'label2id'。
            
        返回:
            Dataset: 處理後的數據集，適合用於訓練或測試。
        """
        tokenizer = kwargs.get('tokenizer')
        label2id = kwargs.get('label2id', {})
        
        if not tokenizer or not label2id:
            raise ValueError("準備NER數據集需要提供 tokenizer 和 label2id 參數。")
        
        tokenized_inputs = {'input_ids': [], 'attention_mask': [], 'labels': []}
        
        for entry in data:
            words = entry['words']
            ner_tags = entry['ner']
            
            # 對詞彙進行分詞，確保正確處理子詞
            encoded = tokenizer(words, is_split_into_words=True, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
            input_ids = encoded['input_ids'][0].tolist()
            attention_mask = encoded['attention_mask'][0].tolist()
            
            # 將NER標籤轉換為ID，處理子詞標記
            labels = []
            word_ids = encoded.word_ids(batch_index=0)  # 獲取每個token對應的word索引
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:  # 特殊token，如[CLS], [SEP], [PAD]
                    labels.append(-100)  # 忽略損失計算
                elif word_idx != previous_word_idx:  # 新的word開始
                    if word_idx < len(ner_tags):
                        tag = ner_tags[word_idx]
                        labels.append(label2id.get(tag, label2id.get('O', 0)))  # 如果標籤未找到，預設為 'O'
                    else:
                        labels.append(label2id.get('O', 0))
                else:  # 子詞token，與前一個word相同標籤
                    if word_idx < len(ner_tags):
                        tag = ner_tags[word_idx]
                        labels.append(label2id.get(tag, label2id.get('O', 0)))
                    else:
                        labels.append(label2id.get('O', 0))
                previous_word_idx = word_idx
            
            tokenized_inputs['input_ids'].append(input_ids)
            tokenized_inputs['attention_mask'].append(attention_mask)
            tokenized_inputs['labels'].append(labels)
        
        dataset = Dataset.from_dict(tokenized_inputs)
        print(f"NER數據集已準備好，包含 {len(dataset)} 個樣本。")
        return dataset

    def get_unique_labels(self, data: List[Dict[str, Any]], **kwargs) -> List[str]:
        """
        從NER數據集中提取唯一的標籤。
        
        參數:
            data (List[Dict[str, Any]]): 數據列表。
            **kwargs: 其他參數。
            
        返回:
            List[str]: 排序後的唯一標籤列表。
        """
        labels = set()
        for entry in data:
            for tag in entry['ner']:
                labels.add(tag)
        unique_labels = sorted(list(labels))
        print(f"提取了 {len(unique_labels)} 個唯一NER標籤。")
        return unique_labels
