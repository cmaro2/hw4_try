import reader
import numpy as np


if __name__ == '__main__':
    with open('log/p2_result.txt', "r") as f:
    #with open('log/p1_valid.txt', "r") as f:
        label = [int(line.strip()) for line in f.readlines()]

    collection = reader.getVideoList('hw4_data/TrimmedVideos/label/gt_valid.csv')
    act_label = collection['Action_labels']
    length = len(act_label)
    cor = 0
    for i in range(length):
        if label[i] == int(act_label[i]):
            cor += 1
    acc = cor/length

    print('Accuracy:', acc)
