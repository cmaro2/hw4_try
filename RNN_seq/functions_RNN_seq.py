import torch


def features(dataset, name=''):

    train_v = []
    train_f = []
    print('Starting Feature extractor of train videos')
    with torch.no_grad():
        for i, (vid, act) in enumerate(dataset):
            train_v.append(vid.squeeze())
            train_f.append(act)

    print('Finished Feature extractor of train videos')

    torch.save(train_v, 'features_RNN_seq'+name+'.pt')
    torch.save(train_f, 'act_RNN_seq'+name+'.pt')
    return train_v, train_f
