import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.functional import accuracy
from config import *
from model import ModelMultipleLabel


def collate_fn(batch):
    input_ids, attention_masks, intents, categories = [], [], [], []
    for text, intent, category in batch:
        encoded_dict = model.tokenizer.encode_plus(
            text,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=MAX_LENGTH,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

        intents.append(model.intents_dict[intent])
        categories.append(model.categories_dict[category])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    intents = torch.tensor(intents, dtype=torch.long)
    categories = torch.tensor(categories, dtype=torch.long)

    return input_ids, attention_masks, intents, categories


def emr(out1, out2, y1, y2):
    out1 = torch.argmax(out1, -1)
    out2 = torch.argmax(out2, -1)

    return ((out1 == y1) * (out2 == y2)).sum() / y2.shape[0]


class CustomData(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, -3]
        intent = self.df.iloc[idx, -2]
        category = self.df.iloc[idx, -1]
        if self.transforms:
            text = self.transforms(text=text)["text"]

        return text, intent, category


class LitClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.train_acc1 = torchmetrics.Accuracy()
        self.train_acc2 = torchmetrics.Accuracy()
        self.valid_acc1 = torchmetrics.Accuracy()
        self.valid_acc2 = torchmetrics.Accuracy()

    def configure_optimizers(self):
        optimizer = AdamW(model.fc.parameters(),
                          lr=LR,  # args.learning_rate - default is 5e-5,
                          eps=EPS  # args.adam_epsilon  - default is 1e-8.
                          )
        total_steps = len(train_loader) * EPOCH

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x1, x2, y1, y2 = train_batch

        out = model.backbone(x1, x2)[-1]
        out1, out2 = model.fc(out)

        loss1 = F.cross_entropy(out1, y1)
        loss2 = F.cross_entropy(out2, y2)
        loss = ALPHA*loss1 + (1-ALPHA)*loss2

        self.log('train_loss', loss.item(), on_step=False, on_epoch=True)

        acc1 = accuracy(out1, y1)
        acc2 = accuracy(out2, y2)

        self.log('train_acc1', acc1, on_step=False, on_epoch=True)
        self.log('train_acc2', acc2, on_step=False, on_epoch=True)
        self.log("emr_train", emr(out1, out2, y1, y2), on_step=False, on_epoch=True)

        return loss


model = ModelMultipleLabel(
    name_backbone=BACKBONE_NAME,
    model_file=MODEL_FILE,
    intent_file=INTENT_FILE,
    category_file=CATEGORY_FILE
)

df = pd.read_csv(DATA_FILE, index_col=False)

data_train = CustomData(df)
train_loader = DataLoader(
    data_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

print(type(model.backbone))
print(type(model.fc))

trainer = pl.Trainer(max_epochs=EPOCH)
model_lit = LitClassification()
trainer.fit(model_lit, train_loader)

torch.save(model.fc, MODEL_FILE)
