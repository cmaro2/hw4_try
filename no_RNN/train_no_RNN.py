from __future__ import print_function
import random
import numpy as np
import torch
import os
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from no_RNN import data_no_RNN
from no_RNN.parser_2 import arg_parse
from no_RNN.model_no_RNN import Model


if __name__ == '__main__':
    #load args
    args = arg_parse()

    def save_model(model, save_path):
        torch.save(model.state_dict(), save_path)

    # Get random seed
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Load video dataloader
    train_videos = torch.utils.data.DataLoader(data_no_RNN.DATA('train'),
                                             batch_size=1,
                                             num_workers=args.workers,
                                             shuffle=False)

    valid_videos = torch.utils.data.DataLoader(data_no_RNN.DATA('valid'),
                                             batch_size=1,
                                             num_workers=args.workers,
                                             shuffle=False)

    f_size = 2048
    if 0:
        train_f, train_act = features(train_videos)
    else:
        train_f = torch.load('features_CNN.pt').squeeze()
        train_act = torch.load('act_CNN.pt')

    if 0:
        valid_f, valid_act = features(valid_videos, '_valid')
    else:
        valid_f = torch.load('features_CNN_valid.pt').squeeze()
        valid_act = torch.load('act_CNN_valid.pt')

    model = Model(f_size).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss = nn.CrossEntropyLoss()

    model.train()
    vid_len = len(train_f)
    train_loss = []
    val_acc = []
    max_acc = 0

    print('starting training')
    for epoch in range(args.epoch):
        batch_size = args.train_batch
        avg_loss = 0
        perm_index = torch.randperm(vid_len)
        train_fs = train_f[perm_index]
        train_acts = train_act[perm_index]

        for i in range(0, vid_len, batch_size):
            if batch_size != args.train_batch:
                break
            elif i+batch_size >= vid_len:
                break
                #batch_size = vid_len-i-1

            model.zero_grad()
            vids = train_fs[i:i+batch_size].squeeze().cuda()
            acts = train_acts[i:i + batch_size].cuda()
            output = model(vids)
            ls = loss(output,acts)
            ls.backward()
            optimizer.step()
            avg_loss += ls

        print('Epoch: %d/%d\tAverage Loss: %.4f'
              % (epoch, args.epoch,
                 avg_loss / vid_len))

        # Test on the validation videos
        model.eval()
        out = model(valid_f.cuda())
        out_label = torch.argmax(out,1).cpu().data
        acc = np.mean((out_label == valid_act).numpy())
        train_loss.append(avg_loss)
        val_acc.append(acc)
        print('Accuracy on validation:', acc)
        model.train()
        if acc > max_acc:
            save_model(model, os.path.join(args.save_dir, 'CNN_model.pth.tar'))
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
    plt.savefig("CNN_curve.png")
    plt.show()

