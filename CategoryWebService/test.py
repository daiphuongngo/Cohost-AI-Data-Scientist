import urllib.parse
import requests 

# params = {'q': 'Xin Chào'}

# params = urllib.parse.urlencode(params)

# res = requests.get("http://127.0.0.1:5000/?"+params)

# print(res.text)

headers = {'Content-Type':'application/json'}

data_raw = [
        {
        "text": "Xin chào",
        "intent": "vui khong",
        "entities": [],
        "traits": []
        }
    ]
data_raw = str(data_raw)
data=data_raw.encode('utf-8')

res = requests.post("http://127.0.0.1:5000", headers=headers)
print(res.text)