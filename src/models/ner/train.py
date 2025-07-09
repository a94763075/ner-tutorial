import argparse
import os
from typing import Dict, Any
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from src.models.base.train import BaseTrainer
from src.models.ner.data import NERDataProcessor

class NERTrainer(BaseTrainer):
    """
    NER模型的訓練器，繼承自BaseTrainer，實現NER特定的訓練邏輯。
    """
    def __init__(self, model_name: str, experiment_name: str = "NER_Training", run_name: str = None):
        """
        初始化NER訓練器。
        
        參數:
            model_name (str): 預訓練模型的名稱或路徑。
            experiment_name (str): MLflow 實驗的名稱。
            run_name (str, 可選): 當前運行的名稱。
        """
        super().__init__(experiment_name, run_name)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.data_processor = NERDataProcessor()
        self.metrics_calculator = None

    def train(self, train_data_path: str, output_dir: str, **kwargs) -> None:
        """
        使用指定的數據和參數訓練NER模型。
        
        參數:
            train_data_path (str): 訓練數據JSON檔案的路徑。
            output_dir (str): 儲存訓練模型的目錄。
            **kwargs: 其他訓練參數，例如 epochs, batch_size, val_data_path, test_data_path。
        """
        import time
        from sklearn.model_selection import train_test_split
        from src.models.ner.metrics import NERMetrics
        
        # 創建唯一的訓練運行目錄以避免覆蓋之前的模型
        run_id = time.strftime("%Y%m%d_%H%M%S")
        unique_output_dir = os.path.join(output_dir, f"run_{run_id}")
        os.makedirs(unique_output_dir, exist_ok=True)
        print(f"訓練結果將保存至唯一目錄: {unique_output_dir}，以避免覆蓋之前的模型。")
        
        # 開始 MLflow 運行
        self.start_run()
        
        # 從 kwargs 提取訓練參數
        epochs = kwargs.get('epochs', 3)
        batch_size = kwargs.get('batch_size', 16)
        val_data_path = kwargs.get('val_data_path', None)
        test_data_path = kwargs.get('test_data_path', None)
        
        # 記錄訓練參數到 MLflow
        params = {
            'model_name': self.model_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'output_dir': unique_output_dir,
            'val_data_path': val_data_path if val_data_path else 'split from train',
            'test_data_path': test_data_path if test_data_path else 'not provided'
        }
        self.log_parameters(params)
        
        # 載入數據
        train_data = self.data_processor.load_data(train_data_path)
        unique_labels = self.data_processor.get_unique_labels(train_data)
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        # 初始化評估指標計算器
        self.metrics_calculator = NERMetrics(id2label)
        
        # 載入分詞器和模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(unique_labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        
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
        
        # 定義訓練參數
        training_args = TrainingArguments(
            output_dir=unique_output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
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
            metric_for_best_model="f1"
        )
        
        # 初始化訓練器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.metrics_calculator.compute_metrics
        )
        
        # 訓練模型
        train_result = self.trainer.train()
        
        # 記錄訓練結果到 MLflow
        metrics = train_result.metrics
        for key, value in metrics.items():
            self.log_metric(key, value)
        
        # 記錄驗證結果到 MLflow
        val_result = self.trainer.evaluate()
        print(f"驗證結果: {val_result}") # Add this line to inspect val_result
        for key, value in val_result.items():
            # 檢查值是否為 None 或 NaN，如果是則跳過或給予預設值
            if value is None or (isinstance(value, float) and (value != value)): # value != value checks for NaN
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
                self.log_metric(f"test_{key}", value)
            print(f"測試結果: {test_result}")
        
        # 保存模型
        self.save_model(unique_output_dir)
        
        # 結束 MLflow 運行
        self.end_run()
        print(f"模型訓練完成，已保存至 {unique_output_dir}/final_model")

    def save_model(self, output_dir: str) -> None:
        """
        儲存訓練好的模型和分詞器。
        
        參數:
            output_dir (str): 儲存模型的目錄。
        """
        if self.model and self.tokenizer:
            final_model_dir = os.path.join(output_dir, 'final_model')
            self.model.save_pretrained(final_model_dir)
            self.tokenizer.save_pretrained(final_model_dir)
            print(f"模型和分詞器已儲存至 {final_model_dir}")
            # 將最終模型目錄記錄為 MLflow 工件
            self.log_artifact(final_model_dir, artifact_path="final_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用自定義數據訓練NER模型。")
    parser.add_argument("--model_name", type=str, default="ckiplab/bert-base-chinese-ner", 
                        help="預訓練模型名稱或路徑。")
    parser.add_argument("--train_data_path", type=str, 
                        default="NER_model/ner_model/data/10_train.json", 
                        help="訓練數據JSON檔案的路徑。")
    parser.add_argument("--output_dir", type=str, default="./trained_model", 
                        help="保存訓練模型的目錄。")
    parser.add_argument("--epochs", type=int, default=5, help="訓練的輪數。")
    parser.add_argument("--batch_size", type=int, default=16, help="訓練的批次大小。")
    parser.add_argument("--experiment_name", type=str, default="NER_Training", help="MLflow 實驗名稱。")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow 運行名稱，可選。")
    
    args = parser.parse_args()
    
    trainer = NERTrainer(args.model_name, args.experiment_name, args.run_name)
    trainer.train(args.train_data_path, args.output_dir, epochs=args.epochs, batch_size=args.batch_size)
