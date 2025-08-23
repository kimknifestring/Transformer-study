# train.py

import torch
from pathlib import Path
from dataset import ShakespeareDataset
from model import BigramLanguageModel

# 하이퍼파라미터
SCRIPT_DIR = Path(__file__).parent.parent.resolve()
FILE_PATH = SCRIPT_DIR.parent / '데이터셋.txt'
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 3000
LEARNING_RATE = 1e-2
EVAL_INTERVAL = 300
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# 데이터 준비
dataset = ShakespeareDataset(FILE_PATH, BLOCK_SIZE, BATCH_SIZE)
vocab_size = dataset.vocab_size

# 모델 및 옵티마이저 생성
model = BigramLanguageModel(vocab_size)
m = model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 훈련 루프
for iter in range(MAX_ITERS):
    if iter % EVAL_INTERVAL == 0:
        xb, yb = dataset.get_batch('val')
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits, loss = model(xb, yb)
        print(f"step {iter}: val loss {loss.item():.4f}")

    xb, yb = dataset.get_batch('train')
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 훈련된 모델로 텍스트 생성
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print("\n--- 훈련 후 생성된 텍스트 ---")
print(dataset.decode(m.generate(context, max_new_tokens=300)[0].tolist()))