import argparse
import os
import time
from typing import Dict, Any
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorForTokenClassification
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split

from src.models.base.train import BaseTrainer
from src.models.ner.data import NERDataProcessor
from src.models.ner.metrics import NERMetrics


class NERLoraTrainer(BaseTrainer):
    """
    NER模型的LoRA訓練器，繼承自BaseTrainer，實現NER特定的LoRA訓練邏輯。
    適用於macOS環境，不使用量化技術。
    """
    
    def __init__(
        self, 
        model_name: str, 
        experiment_name: str = "NER_LoRA_Training", 
        run_name: str = None,
        # LoRA specific parameters
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: list = None
    ):
        """
        初始化NER LoRA訓練器。
        
        參數:
            model_name (str): 預訓練模型的名稱或路徑。
            experiment_name (str): MLflow 實驗的名稱。
            run_name (str, 可選): 當前運行的名稱。
            lora_r (int): LoRA rank，控制低秩適應的維度
            lora_alpha (int): LoRA scaling parameter
            lora_dropout (float): LoRA dropout rate
            target_modules (list): 目標模塊列表，如果為None則使用預設值
        """
        super().__init__(experiment_name, run_name)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.data_processor = NERDataProcessor()
        self.metrics_calculator = None
        
        # LoRA parameters
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
            "query", "key", "value", "dense",
            "q_proj", "k_proj", "v_proj", "o_proj"
        ]
        
        # 設置設備 - macOS 優先使用 MPS，否則使用 CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("使用 Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            print("使用 CPU")

    def _setup_lora_config(self) -> LoraConfig:
        """設置LoRA配置"""
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.TOKEN_CLS,
        )
        
        return lora_config

    def _load_model_with_lora(self, unique_labels: list) -> None:
        """使用LoRA配置載入模型"""
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        # 載入分詞器
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 載入模型
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(unique_labels),
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        
        # 將模型移到適當的設備
        self.model = self.model.to(self.device)
        
        # 設置LoRA
        lora_config = self._setup_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        # 啟用梯度檢查點以節省記憶體（如果支援）
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # 列印可訓練參數
        if hasattr(self.model, 'print_trainable_parameters'):
            self.model.print_trainable_parameters()
        else:
            # 手動計算可訓練參數
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.model.parameters())
            print(f"可訓練參數: {trainable_params:,} / 總參數: {all_params:,} "
                  f"({100 * trainable_params / all_params:.2f}%)")
        
        print(f"模型已載入: {self.model_name}")
        print(f"標籤數量: {len(unique_labels)}")
        print(f"使用LoRA配置: r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
        print(f"設備: {self.device}")

    def train(self, train_data_path: str, output_dir: str, **kwargs) -> None:
        """
        使用指定的數據和參數訓練NER模型（LoRA）。
        
        參數:
            train_data_path (str): 訓練數據JSON檔案的路徑。
            output_dir (str): 儲存訓練模型的目錄。
            **kwargs: 其他訓練參數，例如 epochs, batch_size, val_data_path, test_data_path。
        """
        # 創建唯一的訓練運行目錄以避免覆蓋之前的模型
        run_id = time.strftime("%Y%m%d_%H%M%S")
        unique_output_dir = os.path.join(output_dir, f"lora_run_{run_id}")
        os.makedirs(unique_output_dir, exist_ok=True)
        print(f"LoRA訓練結果將保存至唯一目錄: {unique_output_dir}，以避免覆蓋之前的模型。")
        
        # 開始 MLflow 運行
        self.start_run()
        
        # 從 kwargs 提取訓練參數
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 16)  # macOS 可以使用稍大的 batch size
        val_data_path = kwargs.get('val_data_path', None)
        test_data_path = kwargs.get('test_data_path', None)
        learning_rate = kwargs.get('learning_rate', 5e-5)  # 標準 LoRA 學習率
        
        # 記錄訓練參數到 MLflow
        params = {
            'model_name': self.model_name,
            'training_method': 'LoRA',
            'device': str(self.device),
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'output_dir': unique_output_dir,
            'val_data_path': val_data_path if val_data_path else 'split from train',
            'test_data_path': test_data_path if test_data_path else 'not provided',
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout
        }
        self.log_parameters(params)
        
        # 載入數據
        train_data = self.data_processor.load_data(train_data_path)
        unique_labels = self.data_processor.get_unique_labels(train_data)
        
        # 初始化評估指標計算器
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        self.metrics_calculator = NERMetrics(id2label)
        
        # 載入模型with LoRA
        self._load_model_with_lora(unique_labels)
        
        # 準備數據集
        if val_data_path:
            train_dataset = self.data_processor.prepare_dataset(
                train_data, 
                tokenizer=self.tokenizer, 
                label2id=label2id
            )
            val_data = self.data_processor.load_data(val_data_path)
            val_dataset = self.data_processor.prepare_dataset(
                val_data, 
                tokenizer=self.tokenizer, 
                label2id=label2id
            )
        else:
            # 如果沒有提供驗證數據，則從訓練數據中分割
            train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
            train_dataset = self.data_processor.prepare_dataset(
                train_data, 
                tokenizer=self.tokenizer, 
                label2id=label2id
            )
            val_dataset = self.data_processor.prepare_dataset(
                val_data, 
                tokenizer=self.tokenizer, 
                label2id=label2id
            )
        
        # 定義訓練參數 (針對macOS優化)
        training_args = TrainingArguments(
            output_dir=unique_output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=1000,
            save_total_limit=2,
            do_eval=True,
            eval_steps=500,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            # macOS specific settings
            gradient_checkpointing=True if hasattr(self.model, 'gradient_checkpointing_enable') else False,
            # 避免在 MPS 上使用 fp16，因為可能不穩定
            fp16=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            # 使用標準的 AdamW 優化器
            optim="adamw_torch",
            gradient_accumulation_steps=1,
            # 如果使用 MPS，設置適當的設備
            use_mps_device=True if str(self.device) == "mps" else False,
        )
        
        # 數據整理器
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # 初始化訓練器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.metrics_calculator.compute_metrics
        )
        
        # 訓練模型
        print("開始LoRA訓練...")
        train_result = self.trainer.train()
        
        # 記錄訓練結果到 MLflow
        metrics = train_result.metrics
        for key, value in metrics.items():
            self.log_metric(key, value)
        
        # 記錄驗證結果到 MLflow
        val_result = self.trainer.evaluate()
        print(f"驗證結果: {val_result}")
        for key, value in val_result.items():
            # 檢查值是否為 None 或 NaN，如果是則跳過或給予預設值
            if value is None or (isinstance(value, float) and (value != value)):
                print(f"警告: 指標 'val_{key}' 的值為無效，將跳過記錄。")
                continue
            self.log_metric(f"val_{key}", value)
        
        # 如果有測試數據，進行測試並記錄結果
        if test_data_path:
            test_data = self.data_processor.load_data(test_data_path)
            test_dataset = self.data_processor.prepare_dataset(
                test_data, 
                tokenizer=self.tokenizer, 
                label2id=label2id
            )
            test_result = self.trainer.evaluate(test_dataset)
            for key, value in test_result.items():
                if value is None or (isinstance(value, float) and (value != value)):
                    print(f"警告: 指標 'test_{key}' 的值為無效，將跳過記錄。")
                    continue
                self.log_metric(f"test_{key}", value)
            print(f"測試結果: {test_result}")
        
        # 保存模型
        self.save_model(unique_output_dir)
        
        # 結束 MLflow 運行
        self.end_run()
        print(f"LoRA模型訓練完成，已保存至 {unique_output_dir}/final_model")

    def save_model(self, output_dir: str) -> None:
        """
        儲存訓練好的LoRA適配器和分詞器。
        
        參數:
            output_dir (str): 儲存模型的目錄。
        """
        if self.model and self.tokenizer:
            final_model_dir = os.path.join(output_dir, 'final_model')
            
            # 保存LoRA適配器
            self.model.save_pretrained(final_model_dir)
            self.tokenizer.save_pretrained(final_model_dir)
            base_cfg_path = os.path.join(final_model_dir, "config.json")
            self.model.base_model.config.to_json_file(base_cfg_path)            
            # 保存基礎模型的配置信息
            with open(os.path.join(final_model_dir, 'base_model_info.txt'), 'w') as f:
                f.write(f"Base model: {self.model_name}\n")
                f.write(f"LoRA r: {self.lora_r}\n")
                f.write(f"LoRA alpha: {self.lora_alpha}\n")
                f.write(f"LoRA dropout: {self.lora_dropout}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Target modules: {self.target_modules}\n")
            
            print(f"LoRA模型和分詞器已儲存至 {final_model_dir}")
            print("注意：這裡只保存了LoRA適配器，需要與原始基礎模型結合使用。")
            
            # 將最終模型目錄記錄為 MLflow 工件
            self.log_artifact(final_model_dir, artifact_path="final_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用LoRA技術訓練NER模型（macOS版本）。")
    parser.add_argument("--model_name", type=str, default="ckiplab/bert-base-chinese-ner", 
                        help="預訓練模型名稱或路徑。")
    parser.add_argument("--train_data_path", type=str, 
                        default="NER_model/ner_model/data/training.json", 
                        # default="NER_model/ner_model/data/10_train.json", 
                        help="訓練數據JSON檔案的路徑。")
    parser.add_argument("--output_dir", type=str, default="./trained_model", 
                        help="保存訓練模型的目錄。")
    parser.add_argument("--val_data_path", type=str, 
                        default="NER_model/ner_model/data/validation.json", 
                        help="驗證數據JSON檔案的路徑。")
    parser.add_argument("--test_data_path", type=str, 
                        default="NER_model/ner_model/data/testing.json", 
                        help="測試數據JSON檔案的路徑。")
    parser.add_argument("--epochs", type=int, default=5, help="訓練的輪數。")
    parser.add_argument("--batch_size", type=int, default=16, help="訓練的批次大小。")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="學習率。")
    parser.add_argument("--experiment_name", type=str, default="NER_LoRA_Training", 
                        help="MLflow 實驗名稱。")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow 運行名稱，可選。")
    
    # LoRA specific arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank。")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha參數。")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout率。")
    
    args = parser.parse_args()
    
    trainer = NERLoraTrainer(
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    trainer.train(
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_data_path=args.val_data_path,
        test_data_path=args.test_data_path
    )
