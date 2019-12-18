from __future__ import print_function
import random
import numpy as np
import torch
import os
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from RNN import data_RNN
from RNN.parser_2 import arg_parse
from RNN.functions_RNN import features
from RNN.model_RNN import Model

if __name__ == '__main__':
    #load args
    args = arg_parse()

    def save_model(model, save_path):
        torch.save(model.state_dict(), save_path)


    def single_batch_padding(train_X_batch, train_y_batch, test=False):
        if test:
            padded_sequence = nn.utils.rnn.pad_sequence(train_X_batch)
            label = torch.LongTensor(train_y_batch)
            length = [len(train_X_batch[0])]
        else:
            length = [len(x) for x in train_X_batch]
            perm_index = np.argsort(length)[::-1]

            # sort by sequence length
            train_X_batch = [train_X_batch[i] for i in perm_index]
            length = [len(x) for x in train_X_batch]
            padded_sequence = nn.utils.rnn.pad_sequence(train_X_batch)
            label = torch.LongTensor(np.array(train_y_batch)[perm_index])
        return padded_sequence, label, length

    # Get random seed
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    f_size = 2048

    # Obtain features or load them if they are already obtained
    if 0:
        train_videos = torch.utils.data.DataLoader(data_RNN.DATA('train'),
                                                   batch_size=1,
                                                   num_workers=args.workers,
                                                   shuffle=False)
        train_f, train_act = features(train_videos)
    else:
        train_f = torch.load('features_RNN.pt')
        train_act = torch.load('act_RNN.pt')

    if 0:
        valid_videos = torch.utils.data.DataLoader(data_RNN.DATA('valid'),
                                                   batch_size=1,
                                                   num_workers=args.workers,
                                                   shuffle=False)

        valid_f, valid_act = features(valid_videos, '_valid')
    else:
        valid_f = torch.load('features_RNN_valid.pt')
        valid_act = torch.load('act_RNN_valid.pt')

    model = Model(f_size).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss = nn.CrossEntropyLoss()
    model.train()
    vid_len = len(train_f)
    eval_len = len(valid_f)
    train_loss = []
    val_acc = []
    max_acc = 0

    print('starting training')
    for epoch in range(args.epoch):
        batch_size = args.train_batch
        avg_loss = 0

        for i in range(0, vid_len, batch_size):
            if batch_size != args.train_batch:
                break
            elif i + batch_size >= vid_len:
                #break
                batch_size = vid_len-i-1

            model.zero_grad()
            vids = train_f[i:i + batch_size]
            acts = train_act[i:i + batch_size]

            vid_patch, act_patch, length = single_batch_padding(vids, acts)
            vid_patch = vid_patch.cuda()

            output, _ = model(vid_patch.squeeze(), length)
            ls = loss(output, act_patch.cuda())
            ls.backward()
            optimizer.step()
            avg_loss += ls.cpu()

        print('Epoch: %d/%d\tAverage Loss: %.4f'
              % (epoch, args.epoch,
                 avg_loss / vid_len))

        # Test on the validation videos
        model.eval()
        acc = 0
        with torch.no_grad():
            for i in range(0, eval_len, 200):
                vid_e = valid_f[i:i + 200]
                label_e = valid_act[i:i + 200]
                vid_patch, act_patch, length = single_batch_padding(vid_e, label_e)
                vid_patch = vid_patch.squeeze()
                out, _ = model(vid_patch.cuda(), length)
                out_label = torch.argmax(out, 1).cpu().data
                acc += np.sum((out_label == act_patch).numpy())
        acc = acc/eval_len
        train_loss.append(avg_loss)
        val_acc.append(acc)
        print('Accuracy on validation:', acc)
        model.train()
        if acc > max_acc:
            save_model(model, os.path.join(args.save_dir, 'RNN_model.pth.tar'))
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
    plt.savefig("RNN_curve.png")
    plt.show()
