# Import lib
from flask import Flask
from flask_restful import Api, Resource
from flask import request
from transformers import AutoTokenizer
import torch
import pickle
import json
from config import *
from model import *

# *****************



# Hàm dự đoán kết quả, với đầu vào là text, kết quả sẽ ra confident 2 loại label
model = ModelMultipleLabel(
    name_backbone=BACKBONE_NAME,
    model_file=MODEL_FILE,
    intent_file=INTENT_FILE,
    category_file=CATEGORY_FILE
)
##########################


app = Flask(__name__)
api = Api(app)


class CategoryWeb(Resource):
    # Thêm phương phức get với tham số q
    def get(self):
        if "q" in request.args:
            text = request.args["q"]
            return model.predict(text)


api.add_resource(CategoryWeb, "/")

if __name__ == "__main__":
    app.run()
