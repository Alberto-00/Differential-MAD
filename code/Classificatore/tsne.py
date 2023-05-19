from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from numpy import reshape
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

iris = pd.read_csv("C:\\Users\\salva\\Desktop\\Tirocinio\\codicenostro\\Curvature2.csv", sep=";")
y = iris["Etichetta"]
x = iris.drop(["Etichetta","Nome immagine"], axis=1)
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(x)
tsne = TSNE(n_components=2, verbose=1, perplexity=3)
z = tsne.fit_transform(pca_result_50)
iris["y"] = y
iris["Curvature"] = z[:,0]
iris["Etichetta"] = z[:,1]

sns.scatterplot(x="Curvature", y="Etichetta", hue=iris.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=iris).set(title="Curvature")

plt.show()
