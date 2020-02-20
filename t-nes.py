from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = 'SVG'
#X为特征向量，num为向量个数
def vis(X,num):
    # 参数高维可视化
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(X)
    plt.figure(figsize=(18, 9))
    plt.scatter(tsne[:num, 0], tsne[:num, 1],c = "blue")
    plt.colorbar()  # 使用这一句就可以分辨出，颜色对应的类了
    plt.savefig("show.svg", format="svg")
    plt.show()
