import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import plotly.express as px
import json
import plotly
from flask import Flask, render_template

# Load data để lấy các câu nói của người dùng
df = pd.read_csv("data.csv")
sentences = list(df['texts'])

# Load model ở trên huggingFace
PhobertTokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

# tiền sử lý text, tạo thành các đầu vào phù hợp cho model
inputs = PhobertTokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Kết của đầu ra là các vecter 786 chiều ứng với mỗi câu nói 
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output


# Thuật toán phân cụm sử dụng sklearn
# Đầu và các các vecter 786 chiều
# Đầu ra tương ứng với các vecter đó
def clustering(embeddings, num_clusters):
    model_kmean = KMeans(n_clusters=num_clusters)
    model_kmean.fit(embeddings)
    return model_kmean.labels_


# Trực quan hóa dữ liệu sử dụng plotly
# Đầu và là các câu và các vecter 786 chiều
# Đầu ra là đồ thị tương tác của plotly
def visualization(sentences, embeddings):
    X = np.array(embeddings)
    pca = PCA(n_components=3)
    result = pca.fit_transform(X)
    df = pd.DataFrame({
        'sent': sentences,
        'cluster': clustering(embeddings, 6).astype(str),
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
    # fig.show()
    return fig


app = Flask(__name__)


@app.route('/clustering')
def clustering():
    # Mã hóa đồ thị thành dạng json và gửi nó đến template html
    graphJSON = json.dumps(visualization(sentences, embeddings), cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('clustering.html', graphJSON=graphJSON)


if __name__ == "__main__":
    app.run()
