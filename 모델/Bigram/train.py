# train.py

import torch
from pathlib import Path
from dataset import Dataset
from model import BigramLanguageModel
import config

# torch.manual_seed(1337)

# 데이터 준비
dataset = Dataset(config.FILE_PATH, config.BLOCK_SIZE, config.BATCH_SIZE)
vocab_size = dataset.vocab_size

# 모델 및 옵티마이저 생성
model = BigramLanguageModel(vocab_size)
m = model.to(config.DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

# 훈련 루프
for iter in range(1+config.MAX_ITERS+1):
    xb, yb = dataset.get_batch('train')
    xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % config.EVAL_INTERVAL == 0:
        print(f"step {iter}: val loss {loss.item():.4f}")

# 훈련된 모델 가중치 저장
torch.save(model.state_dict(),config.MODEL_PATH)
print(f"모델이 {config.MODEL_PATH}에 저장되었습니다.")
# 훈련된 모델로 텍스트 생성
context = torch.zeros((1, 1), dtype=torch.long, device=config.DEVICE)
print("\n--- 훈련 후 생성된 텍스트 ---")
print(dataset.decode(m.generate(context, max_new_tokens=300)[0].tolist()))