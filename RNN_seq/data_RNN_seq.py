import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import torch
import torchvision

# Set mean and STD values
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class DATA(Dataset):
    def __init__(self, mode='train'):
        # Set parameters to load dataset
        if mode == 'train':
            dir = '../hw4_data/FullLengthVideos/videos/train'
            labels = '../hw4_data/FullLengthVideos/labels/train'
        else:
            dir = '../hw4_data/FullLengthVideos/videos/valid'
            labels = '../hw4_data/FullLengthVideos/labels/valid'

        vid_folders = sorted(os.listdir(dir))
        labels_files = sorted(os.listdir(labels))

        self.all_videos = []
        self.all_labels = []

        for video in vid_folders:
            image_in_folder = sorted(os.listdir(os.path.join(dir, video)))
            video_frames = []
            for frame in image_in_folder:
                frame_dir = os.path.join(dir, video, frame)
                video_frames.append(frame_dir)
            self.all_videos.append(video_frames)

        for label in labels_files:
            lab_dir = os.path.join(labels, label)
            with open(lab_dir, "r") as f:
                label = [int(line.strip()) for line in f.readlines()]
                self.all_labels.append(label)

        self.transform = transforms.Compose([
            transforms.Pad((0, 40), fill=0, padding_mode='constant'),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):

        feature_ext = torchvision.models.resnet50(pretrained=True).cuda()
        feature_ext = nn.Sequential(*list(feature_ext.children())[:-1]).cuda()
        feature_ext.eval()

        # get video and label
        video = self.all_videos[idx]

        trans_video = []
        tam = len(video)
        batch_size = 200
        print('Starting doing resnet for video number ', idx)
        with torch.no_grad():
            for i in range(0, tam, batch_size):
                vid_b = video[i:i + batch_size]
                vid_array = []
                for vid in vid_b:
                    img = Image.open(vid).convert('RGB')
                    vid_array.append(self.transform(img).numpy().astype(np.float))
                vid_array = torch.FloatTensor(vid_array)
                features = feature_ext(vid_array.cuda()).cpu()
                trans_video.append(features.view(-1,2048))
                print('Processed ', i, '/', tam)
            label = self.all_labels[idx]
            trans_video = torch.cat(trans_video, dim=0)

        return trans_video, label
