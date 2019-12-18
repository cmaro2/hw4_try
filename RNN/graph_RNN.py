import torch.utils.data
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from RNN.parser_2 import arg_parse
from RNN.model_RNN import Model
from sklearn.manifold import TSNE
import pandas as pd


if __name__ == '__main__':

    args = arg_parse()

    def single_batch_padding(train_X_batch, train_y_batch, test=False):
        if test == True:
            padded_sequence = nn.utils.rnn.pad_sequence(train_X_batch)
            label = torch.LongTensor(train_y_batch)
            length = [len(train_X_batch[0])]
        else:
            length = [len(x) for x in train_X_batch]
            perm_index = np.argsort(length)[::-1]

            # sort by sequence length
            train_X_batch = [train_X_batch[i] for i in perm_index]
            length = [len(x) for x in train_X_batch]
            padded_sequence = nn.utils.rnn.pad_sequence(train_X_batch)
            label = torch.LongTensor(np.array(train_y_batch)[perm_index])
        return padded_sequence, label, length

    def arrange_data(dataset, lab, model):
        features = np.empty([1, 512])
        labels = np.empty([1])
        b1 = 1000
        batch_size = 1000
        vid_len = len(dataset)
        for i in range(0, vid_len, batch_size):
            if batch_size != b1:
                break
            elif i+batch_size >= vid_len:
                batch_size = vid_len-i-1

            vids = dataset[i:i + batch_size]
            acts = lab[i:i + batch_size]

            vid_patch, act_patch, length = single_batch_padding(vids, acts)
            vid_patch = vid_patch.squeeze().cuda()

            _, feat = model(vid_patch, length)

            if i == 0:
                labels = np.array(act_patch)
                features = feat.cpu().detach().numpy()

            else:
                labels = np.concatenate((labels, np.array(act_patch)))
                features = np.concatenate((features, feat.cpu().detach().numpy()))

        return features, labels


    train_f = torch.load('features_RNN.pt')
    train_act = torch.load('act_RNN.pt')
    model_state = os.path.join(args.save_dir, '../RNN_model.pth.tar')
    my_model = Model(2048).eval().cuda()
    my_model.load_state_dict(torch.load(model_state))
    print('aranging data')
    data, label = arrange_data(train_f, train_act, my_model)

    print('Starting TSNE')
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)

    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    df['y'] = label

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

