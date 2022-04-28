import readline
import pytorch_lightning as pl
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
import torch 
from torch.utils.data import DataLoader
from collections import Counter


class TranslationData:
    def __init__(self, train_src = "train.vi", train_tgt = "train.en"):
        self.train_src = self.load_file(train_src)
        self.train_tgt = self.load_file(train_tgt)
    
    def load_file(self, file_name):
        with open(file_name, "r") as f:
            data = f.readlines()
        return data

    def __len__(self):
        return len(self.train_src)
    
    def __getitem__(self, idx):
        source = self.train_src[idx]
        target = self.train_tgt[idx]

        return source, target


class TranslationDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 2, direction="vien", len_tokens=150):
        super().__init__()
   
        # Define the model
        self.direction = direction
        if direction == "vien":
            self.tokenizer_src = Tokenizer.from_file("tokenizer_vi.json")
            self.tokenizer_tgt = Tokenizer.from_file("tokenizer_en.json")
        elif direction == "envi":
            self.tokenizer_src = Tokenizer.from_file("tokenizer_en.json")
            self.tokenizer_tgt = Tokenizer.from_file("tokenizer_vi.json")
        else:
            raise Exception("Eror")
        
        self.len_tokens = len_tokens
        self.tokenizer_src.enable_padding(length=len_tokens)
        self.tokenizer_tgt.enable_padding(length=len_tokens, direction='left')

        
        self.tokenizer_tgt.post_processor = TemplateProcessing(single="$A")


        # Defining batch size of our data
        self.batch_size = batch_size
        self.direction = direction
        # Defining num_workers
        self.num_workers = num_workers

        # # Defining Tokenizers
        # self.tokenizer = transformer_tokenizer

        # Define label pad token id
        # self.label_pad_token_id = self.tokenizer.token_to_id("[PAD]")
        # self.padding = True
  
    # def prepare_data(self):
    #     self.train_data = TranslationData()
        # self.val_data = datasets['validation']
        # self.test_data = datasets['test']
    def check_len(self, x):
        len_x = [len(i) for i in x]
        print(Counter(len_x))
  
    def setup(self, stage=None):
        # Loading the dataset
        # column_names = self.train_data.column_names
        if self.direction == "vien":
            self.train_dataset = TranslationData(train_src="train.vi", train_tgt="train.en")
        elif self.direction == "envi":
            self.train_dataset = TranslationData(train_src="train.en", train_tgt="train.vi")

        # column_names = self.val_data.column_names
        # self.val_dataset = self.val_data.map(
        #     preprocess_function,
        #     batched=True,
        #     remove_columns=column_names,
        #     desc="Running tokenizer on val dataset",
        # )
        # column_names = self.test_data.column_names
        # self.test_dataset = self.test_data.map(
        #     preprocess_function,
        #     batched=True,
        #     remove_columns=column_names,
        #     desc="Running tokenizer on test dataset",
        # )
    def insert_cls_token(self, texts):
        return ['[CLS] ' + text for text in texts]
    def insert_sep_token(self, texts):
        return  [text + ' [SEP]' for text in texts]
    
    def clip_tokens(self, token_ids_batch, mode="src"):
        if mode=="src":
            # print(len(token_ids[:self.len_tokens]))
            return [token_ids[:self.len_tokens] for token_ids in token_ids_batch]
        elif mode=="tgt":
            return [token_ids[-self.len_tokens:] for token_ids in token_ids_batch]
  
    def custom_collate(self,features):
        ## Pad the Batched data
        sources, targets = [], []
        for source, target in features:
            sources.append(source)
            targets.append(target)
        
        sources = [i.ids for i in self.tokenizer_src.encode_batch(sources)]
        targets_pred = [i.ids for i in self.tokenizer_tgt.encode_batch(self.insert_sep_token(targets))]
        targets = [i.ids for i in self.tokenizer_tgt.encode_batch(self.insert_cls_token(targets))]

        

        sources = self.clip_tokens(sources, mode="src")
        targets = self.clip_tokens(targets, mode="tgt")
        targets_pred = self.clip_tokens(targets_pred, mode="tgt")

        self.check_len(sources)
        self.check_len(targets)
        self.check_len(targets_pred)

        assert len(sources) == len(targets) == len(targets_pred)
        # print(len(sources[0]), len(sources[1]))
        # print(sources)
        sources = torch.as_tensor(sources)
        targets_pred = torch.as_tensor(targets_pred)
        targets = torch.as_tensor(targets)

        return sources, targets, targets_pred
        

        
        # self.tokenizer_src.encode_batch

        # label_name = "labels"
        # labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # # same length to return tensors.
        # if labels is not None:
        #     max_label_length = max(len(l) for l in labels)
        #     padding_side = self.tokenizer.padding_side
        #     for feature in features:
        #         remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
        #         feature["labels"] = (
        #             feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
        #         )

        # features = self.tokenizer.pad(
        #     features,
        #     padding=self.padding,
        #     return_tensors="pt",
        # )

        # # prepare decoder_input_ids
        # if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
        #     decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
        #     features["decoder_input_ids"] = decoder_input_ids

        return features
        
    def train_dataloader(self):
        #dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        #return DataLoader(train_dataset, sampler=dist_sampler, batch_size=32)
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    # def val_dataloader(self):
    #      return DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    # def test_dataloader(self):
    #      return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)

    # def predict_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.custom_collate)         

if __name__ == "__main__":
    tranlate_module = TranslationDataModule()
    tranlate_module.setup()
    x, y, z = next(iter(tranlate_module.train_dataloader()))
    print(x.shape, y.shape, z.shape)
