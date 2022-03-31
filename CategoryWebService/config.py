import os

DATA_FILE = DATA_FILE = os.getenv('DATA_FILE', 'file_store/data.csv')

INTENT_FILE = os.getenv("INTENT_FILE", "file_store/intents_list.pkl")
CATEGORY_FILE = os.getenv("CATEGORY_FILE", "file_store/categories_list.pkl")

BACKBONE_NAME = os.getenv("BACKBONE_NAME", "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
MODEL_FILE = os.getenv("MODEL_FILE", "file_store/fc.pt")

MAX_LENGTH = 45

# Config for training
BATCH_SIZE = 32
EPOCH = 100
LR = 8e-4
EPS = 1e-8

ALPHA = 0.7





