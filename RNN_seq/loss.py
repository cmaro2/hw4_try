import torch.nn as nn
import torch


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, model_output, groundtruth, lengths):
        criterion = nn.CrossEntropyLoss()
        loss = 0
        batch_size = model_output.size()[0]

        for i in range(batch_size):
            sample_length = lengths[i]
            target = groundtruth[i].type(torch.LongTensor).cuda()
            prediction = model_output[i][:sample_length]
            partial_loss = criterion(prediction, target)
            loss += partial_loss
        loss = loss / batch_size

        return loss
