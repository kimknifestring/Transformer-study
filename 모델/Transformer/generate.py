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
start_context = '\n'
context = torch.tensor([dataset.encode(start_context)], dtype=torch.long, device=config.DEVICE)

print("\n--- 트랜스포머 아키텍쳐로 생성된 텍스트: ---")
generated_output = m.generate(context, max_new_tokens=500)[0].tolist()

# 숫자 ID로 된 결과를 다시 문자로 변환하여 출력합니다.
print(dataset.decode(generated_output))