import os
import json
from transformers import pipeline, AutoConfig, AutoModelForTokenClassification
from peft import PeftModel
from src.models.ner.load import NERLoader
from transformers import AutoTokenizer
from pathlib import Path

def initialize_ner_model(model_path: str):
    is_lora = (Path(model_path) / "adapter_config.json").exists()

    if is_lora:
        # 讀 adapter_config 取得 base model
        adapter_cfg = json.loads(Path(model_path, "adapter_config.json").read_text())
        base_name = adapter_cfg["base_model_name_or_path"]

        # 合併 label2id
        base_cfg  = AutoConfig.from_pretrained(base_name)
        lora_cfg  = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if lora_cfg.label2id:
            base_cfg.label2id = lora_cfg.label2id
            base_cfg.id2label = {int(k): v for k, v in lora_cfg.id2label.items()}
            base_cfg.num_labels = len(base_cfg.label2id)

        # ★ 關鍵：忽略不符尺寸的分類頭
        base_model = AutoModelForTokenClassification.from_pretrained(
            base_name,
            config=base_cfg,
            ignore_mismatched_sizes=True
        )

        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_name)

    else:
        # 檢查model_path是否為本地路徑
        is_local_path = os.path.isdir(model_path)
        
        if is_local_path:
            model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        else:
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 創建pipeline
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    return ner_pipeline

def process_text(text: str, ner_pipeline):
    """
    使用NER pipeline處理文本並返回結果。
    """
    results = ner_pipeline(text)
    # 將numpy.float32轉換為標準的float，使其可以被JSON序列化
    for result in results:
        if "score" in result:
            result["score"] = float(result["score"])
    return results
