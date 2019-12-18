import torch.utils.data
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from no_RNN.parser_2 import arg_parse
from no_RNN.model_no_RNN import Model
from sklearn.manifold import TSNE
import pandas as pd

if __name__ == '__main__':

    train_f = torch.load('features_CNN.pt').squeeze()
    train_act = torch.load('act_CNN.pt')

    print("done")

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(train_f)

    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    df['y'] = train_act

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 11),
        data=df,
        legend="full",
        alpha=0.8
    )
    plt.show()

