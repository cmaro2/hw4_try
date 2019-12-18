from __future__ import print_function
import torch.utils.data
from RNN_seq.data_final import DATA
from RNN_seq.features_final import features
from RNN_seq.model_RNN_seq import Model
from parser_2 import arg_parse
import os


if __name__ == '__main__':
    args = arg_parse()
    valid_videos = torch.utils.data.DataLoader(DATA(args.fullvid_dir),
                                               batch_size=1,
                                               num_workers=1,
                                               shuffle=False)

    valid_f, valid_dir = features(valid_videos)

    model = Model(2048, hidden_size=512, dropout=0.5, n_layers=2).cuda()
    model_state = 'RNN_seq_model.pth.tar'
    model.load_state_dict(torch.load(model_state))
    model.eval()

    length = len(valid_f)

    with torch.no_grad():

        for video, vid_dir in zip(valid_f, valid_dir):
            file = open(os.path.join(args.save_dir, str(vid_dir[0])+'.txt'), "w")
            length = len(video)
            out = model(video.unsqueeze(0).cuda(), [length])
            out_label = torch.argmax(out.squeeze(), 1).cpu().data.numpy()
            for i in range(len(out_label)):
                if i == len(out_label) - 1:
                    file.write(str(out_label[i]))
                else:
                    file.write(str(out_label[i]) + '\n')
            file.close()
