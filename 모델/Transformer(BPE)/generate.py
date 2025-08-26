import torch
import config
from model import TransformerLanguageModel
from dataset import Dataset

# 데이터 준비
dataset = Dataset(config.FILE_PATH, config.BLOCK_SIZE, config.BATCH_SIZE)
vocab_size = dataset.vocab_size

# 모델 불러오기
model = TransformerLanguageModel(vocab_size)
m = model.to(config.DEVICE)

# 저장된 가중치(state_dict) 불러오기
print(f"Loading model from {config.MODEL_PATH}...")
m.load_state_dict(torch.load(config.MODEL_PATH, weights_only=True))
m.eval() # 모델을 평가 모드로 설정
print("Model loaded successfully.")

# 텍스트 생성
# start_context = '\n'
start_context = input("예측할 문장을 입력:")
context = torch.tensor([dataset.encode(start_context)], dtype=torch.long, device=config.DEVICE)

print("\n--- 트랜스포머 아키텍쳐로 생성된 텍스트: ---")
print(start_context,end='')
for token_tensor in m.generate(context, max_new_tokens=500):
    new_char = dataset.decode([token_tensor.item()])
    print(new_char, end='', flush=True)

print()