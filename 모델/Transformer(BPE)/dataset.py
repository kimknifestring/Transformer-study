import torch
import config
from tokenizers import Tokenizer
class Dataset:
    def __init__(self, file_path, block_size, batch_size):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.block_size = block_size
        self.batch_size = batch_size

        # 토크나이저 불러오기
        tokenizer_path = config.VOCAB_DIR / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
        self.vocab_size = self.tokenizer.get_vocab_size()

        self.encode = lambda s: self.tokenizer.encode(s).ids
        self.decode = lambda l: self.tokenizer.decode(l)

        # 데이터 분할
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(data)) # 90% train, 10% val
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y