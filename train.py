import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import argparse
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import gc
import psutil
from collections import OrderedDict
import threading
import time
import weakref

# --------- 1. 參數解析 ---------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, required=True, help="Path to training json file")
    parser.add_argument("--eval_json", type=str, required=True, help="Path to eval json file")
    parser.add_argument("--output_dir", type=str, default="./whisper-finetune-output", help="Output dir for model and logs")
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v2", help="Base model to finetune")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size per device")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_audio_len", type=float, default=30.0, help="Maximum audio length in seconds")
    parser.add_argument("--min_audio_len", type=float, default=0.5, help="Minimum audio length in seconds")
    parser.add_argument("--cache_size_gb", type=float, default=8.0, help="Audio cache size in GB")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()

# --------- 2. 輕量級記憶體快取系統 ---------
class LightweightAudioCache:
    def __init__(self, max_size_gb=2.0):  # 大幅降低快取大小
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.cache = OrderedDict()
        self.current_size = 0
        self.lock = threading.Lock()
        self.hit_count = 0
        self.miss_count = 0

    def _estimate_size(self, audio_array):
        return audio_array.nbytes if hasattr(audio_array, 'nbytes') else len(audio_array) * 4

    def _evict_lru(self, needed_size):
        # 更積極的清理策略
        while (self.current_size + needed_size > self.max_size_bytes or
               len(self.cache) > 500) and self.cache:  # 限制項目數量
            oldest_key, oldest_data = self.cache.popitem(last=False)
            self.current_size -= self._estimate_size(oldest_data)
            del oldest_data  # 明確刪除

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.hit_count += 1
                value = self.cache.pop(key)
                self.cache[key] = value
                return value.copy()  # 返回副本避免修改
            self.miss_count += 1
            return None

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                return

            size = self._estimate_size(value)
            # 如果單個文件超過快取大小的10%，直接跳過
            if size > self.max_size_bytes * 0.1:
                return

            self._evict_lru(size)
            self.cache[key] = value.copy()
            self.current_size += size

    def clear(self):
        """清空快取"""
        with self.lock:
            self.cache.clear()
            self.current_size = 0
            gc.collect()

    def get_cache_info(self):
        with self.lock:
            hit_rate = self.hit_count / (self.hit_count + self.miss_count) if (self.hit_count + self.miss_count) > 0 else 0
            return {
                'size': len(self.cache),
                'memory_mb': self.current_size / (1024 * 1024),
                'max_memory_mb': self.max_size_bytes / (1024 * 1024),
                'hit_rate': hit_rate
            }

# 全局快取實例 - 每個進程獨立
audio_cache = None

def load_audio_fast(audio_path):
    """使用torchaudio快速載入音頻，記憶體優化版"""
    global audio_cache

    # 檢查快取
    if audio_cache is not None:
        cached_audio = audio_cache.get(audio_path)
        if cached_audio is not None:
            return cached_audio

    try:
        # 使用torchaudio載入，比librosa快很多
        waveform, sample_rate = torchaudio.load(audio_path)

        # 轉換為單聲道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 重採樣到16kHz (如果需要) - 使用共享的resampler
        if sample_rate != 16000:
            # 創建全局共享的resamplers字典
            if not hasattr(load_audio_fast, '_resamplers'):
                load_audio_fast._resamplers = {}

            if sample_rate not in load_audio_fast._resamplers:
                load_audio_fast._resamplers[sample_rate] = torchaudio.transforms.Resample(
                    sample_rate, 16000
                )

            waveform = load_audio_fast._resamplers[sample_rate](waveform)

        # 轉換為numpy
        audio_array = waveform.squeeze().numpy().astype(np.float32)

        # 嘗試快取 (如果啟用且大小合適)
        if audio_cache is not None:
            audio_cache.put(audio_path, audio_array)

        # 明確清理tensor
        del waveform

        return audio_array

    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return None

# 添加記憶體清理函數
def cleanup_memory():
    """強制清理記憶體"""
    gc.collect()
    # 移除CUDA操作避免多進程錯誤

def prepare_example_minimal(example):
    """最小化預處理，只保留必要信息"""
    try:
        return {
            "audio_path": example["audio"]["path"],
            "sentence": example["sentence"],
            "duration": example["duration"]
        }
    except Exception as e:
        print(f"Error preparing example: {e}")
        return None

# --------- 3. 記憶體優化的資料整理器 ---------
class MemoryOptimizedDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.feature_extractor = processor.feature_extractor
        self.tokenizer = processor.tokenizer
        self.batch_count = 0

    def __call__(self, batch):
        # 過濾None樣本
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        input_features = []
        labels = []

        try:
            # 逐個處理避免記憶體積累
            for item in batch:
                speech_array = load_audio_fast(item["audio_path"])
                if speech_array is None:
                    continue

                # 提取特徵
                features = self.feature_extractor(
                    speech_array,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features[0]

                # 處理標籤
                label_ids = self.tokenizer(
                    item["sentence"],
                    max_length=448,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids[0]

                input_features.append(features)
                labels.append(label_ids)

                # 清理中間變量
                del speech_array

            if not input_features:
                return None

            # 批量填充
            input_features_tensor = torch.stack(input_features)
            labels_tensor = pad_sequence(labels, batch_first=True, padding_value=-100)

            # 清理列表
            del input_features, labels

            # 移除自動清理避免多進程CUDA錯誤
            self.batch_count += 1

            return {
                "input_features": input_features_tensor,
                "labels": labels_tensor
            }

        except Exception as e:
            print(f"Error in data collator: {e}")
            # 發生錯誤時只做基本清理
            gc.collect()
            return None

# --------- 4. 記憶體監控和管理回調 ---------
class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, cache_obj):
        self.cache = cache_obj
        self.train_losses = []
        self.eval_losses = []
        self.memory_warnings = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 記錄loss
            if "loss" in logs:
                self.train_losses.append((logs.get("epoch", 0), logs["loss"]))
            if "eval_loss" in logs:
                self.eval_losses.append((logs.get("epoch", 0), logs["eval_loss"]))

            # 每50步檢查一次記憶體狀態
            if state.global_step % 50 == 0:
                memory_info = psutil.virtual_memory()

                # 顯示記憶體狀態
                cache_info = "No cache" if self.cache is None else self.cache.get_cache_info()

                if isinstance(cache_info, dict):
                    print(f"Step {state.global_step}: "
                          f"Cache: {cache_info['size']} items "
                          f"({cache_info['memory_mb']:.1f}MB, hit rate: {cache_info['hit_rate']:.2f}), "
                          f"System RAM: {memory_info.percent:.1f}% used")
                else:
                    print(f"Step {state.global_step}: {cache_info}, "
                          f"System RAM: {memory_info.percent:.1f}% used")

                # 記憶體警告和處理
                if memory_info.percent > 90:
                    self.memory_warnings += 1
                    print(f"⚠️  HIGH MEMORY WARNING {self.memory_warnings}: {memory_info.percent:.1f}% RAM used")

                    # 清理快取
                    if self.cache is not None:
                        self.cache.clear()
                        print("🗑️  Cache cleared to free memory")

                    # 強制垃圾回收 (避免CUDA操作)
                    gc.collect()
                    print("🧹 Memory cleanup completed")

                    # 記憶體危險時暫停訓練
                    if memory_info.percent > 95:
                        print("🚨 CRITICAL MEMORY USAGE! Pausing training...")
                        control.should_save = True
                        # 不直接停止，讓使用者決定

    def on_train_end(self, args, state, control, **kwargs):
        import matplotlib.pyplot as plt

        # 繪製loss曲線
        if self.train_losses or self.eval_losses:
            plt.figure(figsize=(10, 6))

            if self.train_losses:
                epochs, values = zip(*self.train_losses)
                plt.plot(epochs, values, label="Train Loss", alpha=0.7)

            if self.eval_losses:
                epochs, values = zip(*self.eval_losses)
                plt.plot(epochs, values, label="Eval Loss", linewidth=2)

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title("Training Progress - Large Dataset Optimized")
            plt.savefig(os.path.join(args.output_dir, "loss_curve_optimized.png"), dpi=150)
            plt.close()

# --------- 5. 主程式 ---------
if __name__ == "__main__":
    args = parse_args()

    # 初始化輕量級快取 (大幅降低快取大小)
    cache_size = min(args.cache_size_gb, 50.0)  # 最多50GB
    audio_cache = LightweightAudioCache(max_size_gb=cache_size)

    print(f"Loading datasets with smart caching ({args.cache_size_gb}GB)...")

    # 載入資料集
    train_dataset = load_dataset("json", data_files=args.train_json, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_json, split="train")

    print("Loading processor and model...")
    processor = WhisperProcessor.from_pretrained(args.base_model, language="Chinese", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    print("Filtering datasets by duration...")
    train_dataset = train_dataset.filter(
        lambda x: args.min_audio_len <= x["duration"] <= args.max_audio_len,
        desc="Filtering train data"
    )
    eval_dataset = eval_dataset.filter(
        lambda x: args.min_audio_len <= x["duration"] <= args.max_audio_len,
        desc="Filtering eval data"
    )

    print(f"Dataset sizes - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    print("Preparing datasets (minimal preprocessing)...")
    train_dataset = train_dataset.map(
        prepare_example_minimal,
        num_proc=16,  # 增加並行處理
        desc="Preparing train data"
    )

    eval_dataset = eval_dataset.map(
        prepare_example_minimal,
        num_proc=16,
        desc="Preparing eval data"
    )

    # 清理記憶體
    gc.collect()

    # 高效能訓練參數
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        logging_steps=50,  # 更頻繁的logging
        save_steps=1000,
        save_total_limit=3,
        # --- 修改: 載入eval最佳模型 ---
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Trainer會自動保護best checkpoint不被save_total_limit刪除
        # 所以3個checkpoint = best + 最近2個
        # ---------------------------------
        logging_dir=os.path.join(args.output_dir, "logs"),
        evaluation_strategy="steps",
        eval_steps=1000,
        bf16=True,
        remove_unused_columns=False,
        # 降低workers避免CUDA多進程問題
        dataloader_num_workers=4,  # 降低避免CUDA錯誤
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,  # 降低預取
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        dataloader_drop_last=True,
        ignore_data_skip=True,
        # 記憶體優化
        max_steps=-1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
    )

    # 建立記憶體優化訓練器
    data_collator = MemoryOptimizedDataCollator(processor)
    memory_callback = MemoryMonitorCallback(audio_cache)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        callbacks=[memory_callback]
    )

    print("Starting optimized training for large dataset...")
    print(f"Cache configuration: {args.cache_size_gb}GB, {training_args.dataloader_num_workers} workers")
    print(f"Best model selection: enabled (metric=eval_loss, lower is better)")

    # 開始訓練 (支援從checkpoint恢復)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 訓練結束後 model 已經是 eval_loss 最低的那個 (由 load_best_model_at_end=True 保證)
    print("Saving best model (lowest eval_loss)...")
    trainer.save_model(os.path.join(training_args.output_dir, "best_model"))
    processor.save_pretrained(training_args.output_dir)

    # 顯示最終快取統計
    final_cache_info = audio_cache.get_cache_info()
    print(f"Training completed!")
    print(f"Best model saved to: {os.path.join(training_args.output_dir, 'best_model')}")
    print(f"Final cache stats: {final_cache_info['size']} items, "
          f"{final_cache_info['memory_mb']:.1f}MB used")
