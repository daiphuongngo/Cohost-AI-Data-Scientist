import urllib.parse
import requests
from pprint import pprint

params = {'q': 'Xin Chào'}

params = urllib.parse.urlencode(params)

res = requests.get("http://127.0.0.1:5000/?"+params)

pprint(res.text)
