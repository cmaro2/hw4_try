import torch
import torch.nn as nn
import torchvision


def features(dataset):
    feature_ext = torchvision.models.resnet50(pretrained=True).cuda()
    feature_ext = nn.Sequential(*list(feature_ext.children())[:-1]).cuda()
    feature_ext.eval()
    train_f = []

    num_videos = len(dataset)
    print('Starting Feature extractor of videos')
    with torch.no_grad():
        for i, vid in enumerate(dataset):
            vid = vid.squeeze().float().cuda()
            feature = feature_ext(vid).cpu()
            train_f.append(torch.mean(feature, 0))
            # break
            if i % 100 == 0:
                print('Video ', i, '/', num_videos, ' done.')
    print('Finished Feature extractor of videos')

    train_f = torch.stack(train_f)

    return train_f
