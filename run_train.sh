#!/bin/bash

# 台灣中文 Whisper 微調訓練腳本
# 針對大數據集進行記憶體和 IO 優化

# ===== 使用者設定區 =====
# 請根據你的環境修改以下變數

# GPU 設定
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC=4  # 與 CUDA_VISIBLE_DEVICES 的 GPU 數量一致

# 資料路徑
TRAIN_JSON="./data/train.json"
EVAL_JSON="./data/test.json"

# 模型設定
OUTPUT_DIR="./output"
BASE_MODEL="openai/whisper-large-v2"  # 或指向本地 checkpoint 路徑

# 訓練超參數
LEARNING_RATE=1e-5
BATCH_SIZE=6
NUM_EPOCHS=3
MIN_AUDIO_LEN=0.5
MAX_AUDIO_LEN=29.5

# 大數據集優化參數
CACHE_SIZE_GB=50  # 音頻快取大小 (GB)

# 斷點續訓 (如需要，取消下方註解並填入 checkpoint 路徑)
# RESUME_CKPT="--resume_from_checkpoint ${OUTPUT_DIR}/checkpoint-XXXXX"
RESUME_CKPT=""

# ===== 以下不需修改 =====

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 建立輸出目錄
mkdir -p "$OUTPUT_DIR"

# 記錄系統資源狀況
echo "=== 訓練開始前系統狀況 ==="
echo "記憶體使用情況:"
free -h
echo ""
echo "GPU 狀況:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo ""
echo "硬碟使用情況:"
df -h "$SCRIPT_DIR"
echo ""

# 顯示訓練配置
echo "=== 訓練配置 ==="
echo "訓練資料: $TRAIN_JSON"
echo "評估資料: $EVAL_JSON"
echo "輸出目錄: $OUTPUT_DIR"
echo "基礎模型: $BASE_MODEL"
echo "學習率:   $LEARNING_RATE"
echo "批量大小: $BATCH_SIZE"
echo "訓練輪數: $NUM_EPOCHS"
echo "音頻快取: ${CACHE_SIZE_GB}GB"
echo "GPU 設備: $CUDA_VISIBLE_DEVICES"
echo ""

# 記錄開始時間
START_TIME=$(date)
echo "訓練開始時間: $START_TIME"
echo ""

# 多卡大數據集優化訓練
torchrun --nproc_per_node=$NPROC "${SCRIPT_DIR}/train.py" \
  --train_json "$TRAIN_JSON" \
  --eval_json "$EVAL_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --base_model "$BASE_MODEL" \
  --learning_rate $LEARNING_RATE \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --min_audio_len $MIN_AUDIO_LEN \
  --max_audio_len $MAX_AUDIO_LEN \
  --cache_size_gb $CACHE_SIZE_GB \
  $RESUME_CKPT

# 記錄結束時間和狀態
EXIT_CODE=$?
END_TIME=$(date)

echo ""
echo "=== 訓練完成 ==="
echo "開始時間: $START_TIME"
echo "結束時間: $END_TIME"
echo "退出狀態: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "訓練成功完成！"
    echo ""
    echo "模型保存位置: $OUTPUT_DIR/best_model"
    echo "日誌文件:     $OUTPUT_DIR/logs"
    echo "Loss 曲線:    $OUTPUT_DIR/loss_curve_optimized.png"

    # 顯示最終模型大小
    echo "最終模型大小:"
    du -sh "$OUTPUT_DIR"
else
    echo "訓練失敗，退出碼: $EXIT_CODE"
    echo "請檢查錯誤日誌"
fi

echo ""
echo "=== 最終系統狀況 ==="
echo "記憶體使用:"
free -h
echo ""
echo "GPU 狀況:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
