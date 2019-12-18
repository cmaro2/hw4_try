from __future__ import print_function
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib
from RNN_seq.model_RNN_seq import Model


if __name__ == '__main__':

    valid_f = torch.load('features_RNN_seq_valid.pt')
    valid_act = torch.load('act_RNN_seq_valid.pt')

    video_num = 6

    model = Model(2048, hidden_size=512, dropout=0.5, n_layers=2).cuda()
    model_state = '../RNN_seq_model.pth.tar'
    model.load_state_dict(torch.load(model_state))
    model.eval()

    video = valid_f[video_num]
    labels = valid_act[video_num]

    with torch.no_grad():
        length = len(video)
        out = model(video.unsqueeze(0).cuda(), [length])
        out_label = torch.argmax(out.squeeze(), 1).cpu().data.numpy()

    plt.figure(figsize=(16,4))
    ax = plt.subplot(211)
    colors = plt.cm.get_cmap('tab20', 11).colors
    cmap = matplotlib.colors.ListedColormap([colors[idx] for idx in out_label])

    bounds = [i for i in range(len(out_label))]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                                           norm=norm,
                                           boundaries=bounds,
                                           spacing='proportional',
                                           orientation='horizontal')
    ax.set_ylabel('Prediction')

    ax2 = plt.subplot(212)
    cmap = matplotlib.colors.ListedColormap([colors[idx] for idx in labels])
    bounds = [i for i in range(len(out_label))]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,
                                           norm=norm,
                                           boundaries=bounds,
                                           spacing='proportional',
                                           orientation='horizontal')

    ax2.set_ylabel('GroundTruth')
    plt.savefig("temporal_action_segmentation_v5.png")
    plt.show()
