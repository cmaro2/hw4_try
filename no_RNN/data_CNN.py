from torch.utils.data import Dataset
import reader


class DATA(Dataset):
    def __init__(self, dir_data, label):
        # Set parameters to load dataset

        self.dir = dir_data
        collection = reader.getVideoList(label)

        self.vid_cat = collection['Video_category']
        self.vid_name = collection['Video_name']

    def __len__(self):
        return len(self.vid_cat)

    def __getitem__(self, idx):

        # read video
        video = reader.readShortVideo(self.dir, self.vid_cat[idx], self.vid_name[idx])

        return video
