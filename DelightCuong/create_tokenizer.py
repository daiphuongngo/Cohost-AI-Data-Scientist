# !pip install tokenizers
# from tokenizers import BertWordPieceTokenizer

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer


from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers import pre_tokenizers
from tokenizers.pre_tokenizers import Digits
from tokenizers.pre_tokenizers import Whitespace

def create_tokenizer_vi():
    
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))


    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase()])

    
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
    tokenizer.pre_tokenizer = pre_tokenizer

    
    trainer = WordPieceTrainer(
        vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    files = ["train.vi"]
    tokenizer.train(files, trainer)
    tokenizer.save("tokenizer_vi.json")

    print(tokenizer.encode("Xin chào bạn có khỏe không😁 ?").tokens)

def create_tokenizer_en():
    
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

    
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
    tokenizer.pre_tokenizer = pre_tokenizer

    
    trainer = WordPieceTrainer(
        vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    files = ["train.en"]
    tokenizer.train(files, trainer)
    tokenizer.save("tokenizer_en.json")

    print(tokenizer.encode("Hello, y'all! How are you 😁 ?").tokens)

if __name__ == "__main__":
    create_tokenizer_vi()
    create_tokenizer_en()