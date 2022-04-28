from tokenizers import Tokenizer
from model.transformer import test_transformer

def test_tokenizer():
    tokenizer = Tokenizer.from_file("tokenizer_vi.json")
    print(tokenizer.encode("Xin chào tất cả \n\t.").tokens)

    tokenizer = Tokenizer.from_file("tokenizer_en.json")
    print(tokenizer.encode("Hello Everyoddddne \n\t.").tokens)


test_tokenizer()

test_transformer()