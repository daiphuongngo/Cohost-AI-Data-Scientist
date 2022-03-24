import torch
# from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import plotly.express as px
import json
import plotly
from config import *

def pre_procressing(text, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    return input_ids, attention_mask


class Cluster:
    # def __init__(self)

    def read_data(self, name_df_file):
        df = pd.read_csv(name_df_file)
        sentences = list(df['texts'])
        return sentences
    def clustering(self, embeddings, num_clusters):
        model_kmean = KMeans(n_clusters=num_clusters)
        model_kmean.fit(embeddings)
        return model_kmean.labels_

    def visualization(self, sentences, embeddings):
        X = np.array(embeddings)
        pca = PCA(n_components=3)
        result = pca.fit_transform(X)
        df = pd.DataFrame({
            'sent': sentences,
            'cluster': self.clustering(embeddings, 6).astype(str),
            'x': result[:, 0],
            'y': result[:, 1],
            'z': result[:, 2]
        })
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                            color='cluster', hover_name='sent',
                            range_x=[df.x.min() - 1, df.x.max() + 1],
                            range_y=[df.y.min() - 1, df.y.max() + 1],
                            range_z=[df.z.min() - 1, df.z.max() + 1])

        fig.update_traces(hovertemplate='<b>%{hovertext}</b>')
        return fig

    def embedding(self, sentences, model_backbone, tokenizer):
        input_ids, attention_mask = pre_procressing(sentences, tokenizer)
        embeddings = model_backbone(input_ids=input_ids,
                            attention_mask=attention_mask )[-1]
        return embeddings