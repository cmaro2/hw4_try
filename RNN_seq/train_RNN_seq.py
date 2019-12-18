from __future__ import print_function
import random
import numpy as np
import torch
import os
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from RNN_seq.parser_2 import arg_parse
from RNN_seq.model_RNN_seq import Model
from RNN_seq.loss import Loss


if __name__ == '__main__':

    # load args
    args = arg_parse()

    def save_model(model, save_path):
        torch.save(model.state_dict(), save_path)


    def sort_pad(input_feature, input_lengths, input_labels):
        perm_index = np.argsort(input_lengths)[::-1]
        input_feature = [input_feature[i] for i in perm_index]
        input_labels = [input_labels[i] for i in perm_index]
        input_lengths = sorted(input_lengths, reverse=True)
        input_feature = nn.utils.rnn.pad_sequence(input_feature, batch_first=True)
        return input_feature, input_labels, input_lengths


    def cut_frames(features, labels, size=100, overlap=10):
        a = torch.split(features, size - overlap)
        b = torch.split(torch.Tensor(labels), size - overlap)

        cut_features = []
        cut_labels = []
        for i in range(len(a)):
            if i == 0:
                cut_features.append(a[i])
                cut_labels.append(b[i])
            else:
                cut_features.append(torch.cat((a[i - 1][-overlap:], a[i])))
                cut_labels.append(torch.cat((b[i - 1][-overlap:], b[i])))

        lengths = [len(f) for f in cut_labels]

        return cut_features, cut_labels, lengths

    # Get random seed
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    f_size = 2048

    # Obtain features or load them if they are already obtained
    if 0:
        train_videos = torch.utils.data.DataLoader(data_RNN_seq.DATA('train'),
                                                   batch_size=1,
                                                   num_workers=1,
                                                   shuffle=False)
        train_f, train_act = features(train_videos)
    else:
        train_f = torch.load('features_RNN_seq.pt')
        train_act = torch.load('act_RNN_seq.pt')

    if 0:
        valid_videos = torch.utils.data.DataLoader(data_RNN_seq.DATA('valid'),
                                                   batch_size=1,
                                                   num_workers=1,
                                                   shuffle=False)

        valid_f, valid_act = features(valid_videos, '_valid')
    else:
        valid_f = torch.load('features_RNN_seq_valid.pt')
        valid_act = torch.load('act_RNN_seq_valid.pt')

    print('Creating cut frames')
    train_cut_features = []
    train_cut_labels = []
    train_cut_lengths = []
    for category_frames, category_labels in zip(train_f, train_act):
        features, labels, lengths = cut_frames(category_frames, category_labels,size=300, overlap=30)
        train_cut_features += features
        train_cut_labels += labels
        train_cut_lengths += lengths

    model = Model(f_size,hidden_size=512,dropout=0.5, n_layers=2).cuda()
    model_state = '../RNN_model.pth.tar'
    model.load_state_dict(torch.load(model_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss = Loss().cuda()
    model.train()
    vid_len = len(train_cut_labels)
    train_loss = []
    val_acc = []
    max_acc = 0

    print('starting training')
    for epoch in range(args.epoch):
        batch_size = args.train_batch
        avg_loss = 0
        perm_index = np.random.permutation(vid_len)
        train_X = [train_cut_features[i] for i in perm_index]
        train_y = [train_cut_labels[i] for i in perm_index]
        train_lengths = np.array(train_cut_lengths)[perm_index]

        for i in range(0, vid_len, batch_size):
            if batch_size != args.train_batch:
                break
            elif i + batch_size >= vid_len:
                #break
                batch_size = vid_len-i-1

            model.zero_grad()
            vids = train_X[i:i + batch_size]
            acts = train_y[i:i + batch_size]
            lengths = train_lengths[i:i + batch_size]

            patch_vids, patch_labs, patch_lengths = sort_pad(vids, lengths, acts)

            patch_vids = patch_vids.cuda()

            output = model(patch_vids, patch_lengths)
            ls = loss(output, patch_labs, patch_lengths)
            ls.backward()
            optimizer.step()
            avg_loss += ls.cpu().data.numpy()

        print('Epoch: %d/%d\tAverage Loss: %.4f'
              % (epoch, args.epoch,
                 avg_loss / vid_len))

        # Test on the validation videos
        comparison = []
        acc = 0
        len_val = 0
        with torch.no_grad():
            model.eval()
            for valid_vid, valid_label in zip(valid_f, valid_act):
                length = len(valid_label)
                out = model(valid_vid.unsqueeze(0).cuda(), [length])
                out_label = torch.argmax(out.squeeze(), 1).cpu().data.numpy()
                acc += np.sum((out_label == np.array(valid_label)))
                len_val += length
        acc = acc/len_val
        print('Accuracy on validation:', acc)
        model.train()
        train_loss.append(avg_loss)
        val_acc.append(acc)
        if acc > max_acc:
            save_model(model, os.path.join(args.save_dir, 'RNN_seq_model.pth.tar'))
            max_acc = acc

    print('Finished training with', args.epoch, 'epochs. Maximum Accuracy on Validation:', max_acc)

    # Graph showing loss and accuracy on validation
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss)
    plt.title("training loss")
    plt.ylabel("cross entropy")
    plt.xlabel("epoch")
    plt.subplot(1, 2, 2)
    plt.plot(val_acc)
    plt.title("validation accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.savefig("RNN_seq_curve.png")
    plt.show()
