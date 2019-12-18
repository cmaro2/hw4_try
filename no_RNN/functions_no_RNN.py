import torch
import torch.nn as nn
import torchvision


def features(dataset, name=''):
    feature_ext = torchvision.models.resnet50(pretrained=True).cuda()
    feature_ext = nn.Sequential(*list(feature_ext.children())[:-1]).cuda()
    feature_ext.eval()
    train_f = []
    train_act = []
    num_videos = len(dataset)
    print('Starting Feature extractor of train videos')
    with torch.no_grad():
        for i, (vid, act) in enumerate(dataset):
            vid = vid.squeeze().float().cuda()
            feature = feature_ext(vid).cpu()
            # feature = feature.view(-1,f_size)
            train_f.append(torch.mean(feature, 0))
            train_act.append(int(act[0]))
            # break
            if i % 100 == 0:
                print('Video ', i, '/', num_videos, ' done.')
    print('Finished Feature extractor of train videos')

    train_f = torch.stack(train_f)
    train_act = torch.LongTensor(train_act)
    torch.save(train_f, 'features_CNN'+name+'.pt')
    torch.save(train_act, 'act_CNN'+name+'.pt')
    return train_f, train_act
