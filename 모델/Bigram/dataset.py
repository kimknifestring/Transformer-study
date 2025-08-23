# dataset.py

import torch

class ShakespeareDataset:
    def __init__(self, file_path, block_size, batch_size, split_ratio=0.9):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.block_size = block_size
        self.batch_size = batch_size

        # 어휘 사전 생성
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

        # 인코더/디코더 정의
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: ''.join([self.itos[i] for i in l])

        # 데이터 분할
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(split_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y