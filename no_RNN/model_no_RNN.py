from torch import nn


class Model(nn.Module):
    def __init__(self, f_size):
        super(Model, self).__init__()

        self.linear1 = nn.Linear(f_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 11)
        self.softmax = nn.Softmax(1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.relu(self.bn_1(self.linear1(x)))
        x = self.relu(self.bn_2(self.linear2(x)))
        x = self.relu(self.bn_3(self.linear3(x)))
        y_pred = self.softmax(self.linear4(x))
        return y_pred
