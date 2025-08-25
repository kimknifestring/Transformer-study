# config.py

import torch
from pathlib import Path


ROOT_DIR = Path(__file__).resolve()
FILE_PATH = ROOT_DIR.parent.parent.parent / '데이터셋.txt'
MODEL_DIR = ROOT_DIR.parent / 'Model'
MODEL_NAME = 'bigram_model.pth'
MODEL_PATH = MODEL_DIR / MODEL_NAME 

# 하이퍼파라미터
BATCH_SIZE = 32
BLOCK_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_ITERS = 30000
LEARNING_RATE = 1e-2
EVAL_INTERVAL = 300