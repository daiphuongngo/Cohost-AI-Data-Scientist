import pickle

import pandas as pd

df = pd.read_csv("file_store/data.csv", index_col=False)

label1_list = sorted(list(set(df.iloc[:,-2])))
label2_list = sorted(list(set(df.iloc[:,-1])))

label1_dir = {k:v for v, k in enumerate(label1_list)}
label2_dir = {k:v for v, k in enumerate(label2_list)}


with open('file_store/intents_list.pkl', "wb") as f:
    pickle.dump(label1_list, f)

with open('file_store/categories_list.pkl', "wb") as f:
    pickle.dump(label2_list, f)
