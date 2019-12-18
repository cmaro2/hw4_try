import torch


def features(dataset):

    train_v = []
    train_f = []
    print('Starting Feature extractor of train videos')
    with torch.no_grad():
        for i, (vid, act) in enumerate(dataset):
            train_v.append(vid.squeeze())
            train_f.append(act)

    print('Finished Feature extractor of train videos')

    return train_v, train_f
