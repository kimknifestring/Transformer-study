import torch
import torch.nn as nn
from torch.nn import functional as F
import config

class TranformerLangugeModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        # 단어들을 벡터 임베딩
        self.token_embedding_table = nn.Embedding(vocab_size, config.N_EMBD)
        # 문장에서의 위치 정보를 임베딩(이것과 임베딩된 벡터를 잘 사용하면 문맥을 파악하는 것이 가능해짐)
        self.position_embedding_table = nn.Embedding(config.BLOCK_SIZE, config.N_EMBD)
    
    def forward(self, idx, target=None):
        B, T = idx.shape
        # 단어가 가지는 의미
        tok_emb = self.token_embedding_table(idx) # 결과 모양:(B,T,C)
        # 위치가 가지는 의미
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.DEVICE)) # 결과 모양: (T, C)
        # 원랜 형식이 맞지 않아 더할 수 없지만 Pytorch의 브로드캐스팅(작은 텐서를 자동으로 확장하여 큰 텐서와 맞춰 연산이 가능하게 만듬)을 사용하여 더할 수 있음
        x = tok_emb + pos_emb

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        