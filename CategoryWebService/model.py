import os
import pickle

import torch
from transformers import AdamW, AutoModel, AutoTokenizer
from config import *
from utils import *


class FC(torch.nn.Module):
    def __init__(self, out_feat1, out_feat2):
        super().__init__()
        self.fc1 = torch.nn.Linear(768, out_feat1)
        self.fc2 = torch.nn.Linear(768, out_feat2)

    def forward(self, x):
        return self.fc1(x), self.fc2(x)


def get_pickle(name_file):
    with open(name_file, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def get_backbone(name_backbone):
    backbone = AutoModel.from_pretrained(
        name_backbone,
        torchscript=True)

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    backbone.eval()

    return backbone


def get_tokenizer(name_tokenizer):
    tokenizer = AutoTokenizer.from_pretrained(name_tokenizer, use_fast=False)
    return tokenizer


def post_preprocessing(tensor, list_data):
    tensor_confident = torch.softmax(tensor[0], -1)
    tensor_argmax = torch.argmax(tensor_confident, -1).item()
    tensor_confident = tensor_confident[tensor_argmax].item()

    return list_data[tensor_argmax], tensor_confident


def pre_procressing(text, tokenizer):
    encoded_dict = tokenizer.encode_plus(
        text,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=MAX_LENGTH,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    return input_ids, attention_mask


class ModelMultipleLabel:
    def __init__(self,
                 name_backbone,
                 model_file,
                 intent_file,
                 category_file):
        print(name_backbone)
        self.backbone = get_backbone(name_backbone)
        self.tokenizer = get_tokenizer(name_backbone)

        self.intents = get_pickle(intent_file)
        self.categories = get_pickle(category_file)

        if os.path.exists(model_file):
            print(os.path.exists(model_file))
            self.fc = torch.load(model_file)
        else:
            self.fc = FC(
                len(self.intents),
                len(self.categories),
            )

        self.trace_model()

    def trace_model(self):
        self.backbone = torch.jit.trace(self.backbone, self.example_input())

    def example_input(self):
        text = "Xin chào tất cả mọi người"
        return pre_procressing(text, self.tokenizer)

    def predict(self, text):
        input_ids, attention_mask = pre_procressing(text, self.tokenizer)

        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask )[-1]
        intent_out, category_out = self.fc(out)

        intent_name, intent_confident = post_preprocessing(intent_out, self.intents)
        category_name, category_confident = post_preprocessing(category_out, self.categories)

        intent_obj = IntentClass(intent_name, intent_confident)
        category_obj = CategoryClass(category_name, category_confident)

        return ResponseClass(text, intent_obj, category_obj).toJSON()
