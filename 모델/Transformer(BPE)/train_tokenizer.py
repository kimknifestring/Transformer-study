# train_tokenizer.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import config

# 모르는 토큰이 나올 시 '알 수 없다' 라고 지정
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 특수 토큰들을 미리 추가하고 최대 단어는 5000개
trainer = BpeTrainer(vocab_size=5000, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])

tokenizer.pre_tokenizer = Whitespace()
files = [str(config.FILE_PATH)]
tokenizer.train(files, trainer)

tokenizer_path = config.VOCAB_DIR / "tokenizer.json"
tokenizer.save(str(tokenizer_path))

print(f"토크나이저를 학습하고 {tokenizer_path}에 저장했습니다.")