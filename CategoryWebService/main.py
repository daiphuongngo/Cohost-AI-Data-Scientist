
# Import lib
from flask import Flask
from flask_restful import Api, Resource
from flask import request, jsonify
from transformers import AutoTokenizer
import torch 
import pandas as pd
from pprint import pprint
import pickle
# *****************


# Load danh sách các label 
with open('intents_list.pkl', 'rb') as f:
    label1_list = pickle.load(f)

with open('categories_list.pkl', 'rb') as f:
    label2_list = pickle.load(f)

label1_dir = {k:v for v, k in enumerate(label1_list)}
label2_dir = {k:v for v, k in enumerate(label2_list)}
# *******************************

# Load model và bằng cách sử 
tokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
model = torch.jit.load("traced_bert_simcse.pt")
######################

# Hàm sử lý text: text sẽ được mã hóa thành input_id, attention_mask
    # input_id : là các id tương ứng với các mã token
    # attention_mask: là hệ sô chú ý, được sử dụng khi training.
def preprocessing_input(text):
    encoded_dict = tokenizer.encode_plus(
                text,                      
                add_special_tokens = True,
                max_length = 20,           
                pad_to_max_length = True,
                return_attention_mask = True,   
                return_tensors = 'pt',     
            )
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    return input_id, attention_mask
#######################

# Hàm dự đoán kết quả, với đầu vào là text, kết quả sẽ ra confident 2 loại label
def predict(text):
    # Sử lý input
    input_id, attention_mask = preprocessing_input(text)

    # Dự đoán kết quả, out1, out1 là output của 2 loại label
    with torch.no_grad():
        out1, out2 = model(input_id, attention_mask)

    out1_confident = torch.softmax(out1[0], -1) # Chuyển sang phân phối xác xuất lớp
    out1_argmax = torch.argmax(out1_confident, -1).item() # Chọn lớp có xác xuất cao nhất
    out1_confident = out1_confident[out1_argmax].item() # Lấy độ tin cậy của lớp đó

    out2_confident = torch.softmax(out2[0], -1) # # Chuyển sang phân phối xác xuất lớp
    out2_argmax = torch.argmax(out2_confident, -1).item() # Chọn lớp có xác xuất cao nhất
    out2_confident = out2_confident[out2_argmax].item() # Lấy độ tin cậy của lớp đó

    confident_label = out2_confident*out1_confident

    return {
        "text": text,
        "intent": {
            "name": label1_list[out1_argmax],
            "confident": out1_confident,
        },
        "category": {
            "name": label2_list[out2_argmax],
            "confident": out2_confident,
        },
        "confident_label": confident_label,
    }


app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    # Thêm phương phức get với tham số q
    def get(self):
        if "q" in request.args:
            text = request.args["q"]
            return predict(text)
    # def post(self):
    #     return predict(text)

api.add_resource(HelloWorld, "/")




if __name__ == "__main__":
	app.run()