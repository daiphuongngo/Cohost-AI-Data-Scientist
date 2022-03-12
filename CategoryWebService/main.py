# Import lib
from flask import Flask
from flask_restful import Api, Resource
from flask import request
from transformers import AutoTokenizer
import torch
import pickle
import json
from config import *

# *****************


# Load danh sách các label 
with open(INTENT_FILE, 'rb') as f:
    label1_list = pickle.load(f)

with open(CATEGORY_FILE, 'rb') as f:
    label2_list = pickle.load(f)

label1_dir = {k: v for v, k in enumerate(label1_list)}
label2_dir = {k: v for v, k in enumerate(label2_list)}
# *******************************

# Load model và bằng cách sử 
tokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
model = torch.jit.load(MODEL_FILE)


######################

# Hàm sử lý text: text sẽ được mã hóa thành input_id, attention_mask
# input_id : là các id tương ứng với các mã token
# attention_mask: là hệ sô chú ý, được sử dụng khi training.
def preprocessing_input(text):
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=20,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    return input_id, attention_mask


#######################
class IntentClass:
    def __init__(self, name, confident):
        self.name = name
        self.confident = confident


class CategoryClass:
    def __init__(self, name, confident):
        self.name = name
        self.confident = confident


class ResponseClass:
    def __init__(self, text, intentObj, categoryObj):
        self.text = text
        self.intent = intentObj
        self.category = categoryObj

    def toJSON(self):
        return json.loads(json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4))


# Hàm dự đoán kết quả, với đầu vào là text, kết quả sẽ ra confident 2 loại label
def predict(text):
    # Sử lý input
    input_id, attention_mask = preprocessing_input(text)

    # Dự đoán kết quả, out1, out1 là output của 2 loại label
    with torch.no_grad():
        out1, out2 = model(input_id, attention_mask)
        # out1 là vecter đại diện cho intent
        # out2 là vecter đại diện cho category

    out1_confident = torch.softmax(out1[0], -1)  # Chuyển sang phân phối xác xuất lớp
    out1_argmax = torch.argmax(out1_confident, -1).item()  # Chọn lớp có xác xuất cao nhất
    out1_confident = out1_confident[out1_argmax].item()  # Lấy độ tin cậy của lớp đó

    out2_confident = torch.softmax(out2[0], -1)  # # Chuyển sang phân phối xác xuất lớp
    out2_argmax = torch.argmax(out2_confident, -1).item()  # Chọn lớp có xác xuất cao nhất
    out2_confident = out2_confident[out2_argmax].item()  # Lấy độ tin cậy của lớp đó

    # confident_label = out2_confident * out1_confident

    # Trả về json
    intentObj = IntentClass(label1_list[out1_argmax], out1_confident)
    categoryObj = CategoryClass(label2_list[out2_argmax], out2_confident)
    return ResponseClass(text, intentObj, categoryObj).toJSON()
##########################


app = Flask(__name__)
api = Api(app)


class CategoryWeb(Resource):
    # Thêm phương phức get với tham số q
    def get(self):
        if "q" in request.args:
            text = request.args["q"]
            return predict(text)


api.add_resource(CategoryWeb, "/")

if __name__ == "__main__":
    app.run()
