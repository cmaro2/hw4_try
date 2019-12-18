from RNN.model_RNN import Model
from no_RNN.data_CNN import DATA
from RNN.RNN_features import features
from parser_2 import arg_parse
import torch
import torch.nn as nn
import numpy as np
import os

if __name__ == '__main__':
    args = arg_parse()


    def single_batch_padding(train_X_batch, test=False):
        if test == True:
            padded_sequence = nn.utils.rnn.pad_sequence(train_X_batch)
            length = [len(train_X_batch)]
        else:
            length = [len(x) for x in train_X_batch]
            perm_index = np.argsort(length)[::-1]

            # sort by sequence length
            train_X_batch = [train_X_batch[i] for i in perm_index]
            length = [len(x) for x in train_X_batch]
            padded_sequence = nn.utils.rnn.pad_sequence(train_X_batch)
        return padded_sequence, length

    model_state = 'RNN_model.pth.tar'
    my_model = Model(2048).eval().cuda()
    my_model.load_state_dict(torch.load(model_state))

    valid_videos = torch.utils.data.DataLoader(DATA(args.dir_vid, args.dir_lab),
                                                 batch_size=1,
                                                 num_workers=1,
                                                 shuffle=False)

    print('Obtaining Features')
    feat = features(valid_videos)

    print('Obtaining Output')
    eval_len = len(feat)
    out_labs = []
    with torch.no_grad():
        for i in range(0, eval_len):
            vid_e = feat[i]
            vid_patch, length = single_batch_padding(vid_e, test=True)
            vid_patch = vid_patch.squeeze().cuda()
            vid_patch = vid_patch.permute(1, 0).unsqueeze(1)
            output, _ = my_model(vid_patch, length)
            output = output.cpu()
            out_label = torch.argmax(output, len(output)).cpu().data
            out_labs = out_labs+(out_label.tolist())


    print('saving data in file')
    file = open(os.path.join(args.save_dir, 'p2_result.txt'), "w")
    for i in range(len(out_labs)):
        if i == len(out_labs)-1:
            file.write(str(out_labs[i]))
        else:
            file.write(str(out_labs[i]) + '\n')
    file.close()

    print('done')
