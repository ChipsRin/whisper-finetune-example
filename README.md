# Whisper-finetune-example

基於 OpenAI Whisper 模型，針對台灣中文語音進行微調訓練。支援大規模資料集（數千至上萬小時），內建記憶體優化與音檔快取機制。

## 環境需求

- Python 3.10+
- PyTorch 2.0+（需支援 bf16）
- NVIDIA GPU

### Python 套件

```bash
pip install transformers datasets torchaudio psutil matplotlib
```

## 資料格式

訓練與評估資料使用 JSON Lines 格式，每行一筆資料：

```json
{"audio": {"path": "/path/to/audio.wav"}, "sentence": "對應的文字轉寫", "duration": 3.2}
```

| 欄位 | 說明 |
|------|------|
| `audio.path` | 音頻檔案的絕對路徑（WAV 格式，任意取樣率皆可，會自動轉為 16kHz） |
| `sentence` | 對應的中文文字 |
| `duration` | 音頻長度（秒），用於過濾過短或過長的樣本 |

範例資料請參考 [`data/example.json`](data/example.json)。

## 使用方式

### 1. 準備資料

將訓練集和測試集分別整理成 `train.json` 和 `test.json`，格式如上述。

### 2. 修改設定

編輯 `run_train.sh` 中的變數：

```bash
TRAIN_JSON="./data/train.json"   # 訓練資料路徑
EVAL_JSON="./data/test.json"     # 評估資料路徑
BASE_MODEL="openai/whisper-large-v2"  # 基礎模型（或本地 checkpoint 路徑）
OUTPUT_DIR="./output"            # 輸出目錄
```

GPU 與超參數也可依需求調整。

### 3. 啟動訓練

```bash
bash run_train.sh
```

預設使用 4 張 GPU 透過 `torchrun` 進行分散式訓練。如需調整 GPU 數量，修改 `CUDA_VISIBLE_DEVICES` 和 `NPROC`。


