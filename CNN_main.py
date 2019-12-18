from no_RNN.model_no_RNN import Model
from no_RNN.data_CNN import DATA
from no_RNN.CNN_features import features
from parser_2 import arg_parse
import torch
import os


if __name__ == '__main__':
    args = arg_parse()

    model_state = 'CNN_model.pth.tar'
    my_model = Model(2048).eval().cuda()
    my_model.load_state_dict(torch.load(model_state))

    valid_videos = torch.utils.data.DataLoader(DATA(args.dir_vid, args.dir_lab),
                                                 batch_size=1,
                                                 num_workers=1,
                                                 shuffle=False)

    print('Obtaining Features')
    feat = features(valid_videos).squeeze()
    print('Obtaining Output')
    eval_len = len(feat)
    out_labs = []
    for i in range(0, eval_len, 200):
        vid_e = feat[i:i + 200]
        output = my_model(vid_e.cuda()).cpu()
        out_label = torch.argmax(output, 1).cpu().data
        out_labs = out_labs+(out_label.tolist())

    print('saving data in file')
    file = open(os.path.join(args.save_dir, 'p1_valid.txt'), "w")
    for i in range(len(out_labs)):
        if i == len(out_labs)-1:
            file.write(str(out_labs[i]))
        else:
            file.write(str(out_labs[i]) + '\n')
    file.close()

    print('done')
