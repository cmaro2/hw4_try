import torchvision.transforms as transforms
from torch.utils.data import Dataset
import reader


class DATA(Dataset):
    def __init__(self, mode='train'):
        # Set parameters to load dataset
        if mode == 'train':
            self.dir = '../hw4_data/TrimmedVideos/video/train'
            label = '../hw4_data/TrimmedVideos/label/gt_train.csv'
        else:
            self.dir = '../hw4_data/TrimmedVideos/video/valid'
            label = '../hw4_data/TrimmedVideos/label/gt_valid.csv'

        collection = reader.getVideoList(label)

        self.act_label = collection['Action_labels']
        self.vid_cat = collection['Video_category']
        self.vid_name = collection['Video_name']

    def __len__(self):
        return len(self.act_label)

    def __getitem__(self, idx):

        # read video
        video = reader.readShortVideo(self.dir, self.vid_cat[idx], self.vid_name[idx])

        # get action label
        act = self.act_label[idx]

        return video, act
