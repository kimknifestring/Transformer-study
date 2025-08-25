# generate.py

import torch
from model import BigramLanguageModel
from dataset import Dataset
from pathlib import Path
import config

dataset = Dataset(config.FILE_PATH,config.BLOCK_SIZE,config.BATCH_SIZE)

vocab_size = dataset.vocab_size

model = BigramLanguageModel(vocab_size)
model.to(config.DEVICE)

# 저장된 가중치(state_dict)를 불러옴
model.load_state_dict(torch.load(config.MODEL_PATH,weights_only=True))

# train 상태 해제(Bigram에서는 의미없지만 습관을 들이기 위해 사용함)
model.eval()
print(f"{config.MODEL_PATH} 파일에서 모델을 성공적으로 불러왔습니다.")

my_text = input("예측할 문장을 입력:")
encoded_text = dataset.encode(my_text)
encoded_tensor = torch.tensor(encoded_text,dtype=torch.long,device=config.DEVICE)
# 2차원 텐서를 기대하는 모델이므로 0번 차원에 크기가 1인 차원을 추가한다
context = encoded_tensor.unsqueeze(0)

# 처음부터 문장을 생성하려면 토큰 리스트의 가장 초기에 위치할 \n, 즉 (1,1)크기의 zero벡터를 텐서를 제공하면 된다
# context = torch.zeros((1, 1), dtype=torch.long, device=config.DEVICE)
print("\n--- 저장된 모델(이상 시집 학습)로 생성된 텍스트 ---")
generated_output = model.generate(context, max_new_tokens=300)[0].tolist()
print(dataset.decode(generated_output))