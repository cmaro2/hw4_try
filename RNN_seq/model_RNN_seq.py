from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, input_size, hidden_size=512, n_layers=2, dropout=0.1):
        super(Model, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, n_layers,
                            dropout=(0 if n_layers == 1 else dropout), bidirectional=False,batch_first=True)
        self.bn_0 = nn.BatchNorm1d(self.hidden_size)
        self.fc_1 = nn.Linear(int(self.hidden_size), 11)
        self.softmax = nn.Softmax(1)

    def forward(self, padded_sequence, input_lengths, hidden=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence,input_lengths,batch_first=True)
        outputs, (hn, cn) = self.lstm(packed, hidden)  # outputs: (batch ,seq_len, hidden_features)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        fc_output = self.fc_1(outputs)
        return fc_output
