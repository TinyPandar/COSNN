import numpy as np
import pandas as pd
import torch
from spikingjelly.activation_based import layer, neuron, surrogate, functional
from torch import nn
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, data, ord=True):
        self.label = data.iloc[:, -1]
        self.data = data.iloc[:, :-1]
        self.label = self.label_trans(self.label)
        self.label = self.alter_label(self.label, ord)

        self.data = np.array(self.data, dtype='float64')
        self.label = torch.tensor(self.label)

    def __getitem__(self, index):

        return torch.tensor(self.data[index], dtype=torch.float).tile(50, 1, 1), torch.tensor(self.label[index],
                                                                                              dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def label_trans(self, label):
        label_ret = []
        # 原始的ECG结果及风险级别的字典
        ecg_risk_levels = {
            0: ["N", "Ne", 'Normal', 'N0', '0', 0],
            1: ["AFIB", "SVTA", "SBR", "BI", "NOD", '1', 1, "BBB", "R", "S"],  # 1: Supraventricular ectopic beats
            2: ["VTLR", "B", "VB", "HGEA", "VER", '2', 2, "V"],  # 2: Ventricular ectopic beats
            3: ["VTHR", "VTTdP", "VFL", "VF", 'MI', '3', 3]
        }

        # 构建通过ECG英文缩写获取风险级别的字典
        ecg_abbr_to_risk = {}
        for risk, ecg_list in ecg_risk_levels.items():
            for ecg_abbr in ecg_list:
                ecg_abbr_to_risk[ecg_abbr] = risk

        # 打印通过ECG英文缩写获取风险级别的字典
        print(ecg_abbr_to_risk)

        for l in label:
            label_ret.append(ecg_abbr_to_risk[l])

        distribute = pd.Series(label_ret)
        print(distribute.value_counts())

        return label_ret

    def alter_label(self, label, ord=True):
        # ordinal
        if ord:
            y_return = []
            for y in label:
                y_bin = []
                for i in range(3):
                    y_bin.append(int(y > i))
                y_return.append(y_bin)
            return y_return
        else:
            # classfication
            y_return = []
            for y in label:
                y_bin = [0, 0, 0, 0]
                y_bin[y] = 1
                y_return.append(y_bin)
            return y_return


class COSNN(nn.Module):

    def __init__(self, in_channels, filter_list, m_blocks, kernel_size, n_classes, T: int, use_cupy=False):
        super(COSNN, self).__init__()

        self.in_channels = in_channels
        self.m_blocks = m_blocks
        self.kernel_size = kernel_size
        self.filter_list = filter_list
        self.T = T

        # first conv
        self.conv1 = layer.Conv1d(in_channels=1, out_channels=48, kernel_size=kernel_size * 2, dilation=2,
                                  padding="same")
        self.act1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.maxpool1 = layer.MaxPool1d(kernel_size=2, stride=2)

        # second conv
        self.conv2 = layer.Conv1d(in_channels=48, out_channels=48, kernel_size=kernel_size, dilation=2, groups=2,
                                  padding="same")
        self.act2 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.maxpool2 = layer.MaxPool1d(kernel_size=2, stride=2)

        # classifier
        self.fcP = layer.Linear(180, 1)
        self.fcO = layer.Linear(48, n_classes)
        self.output_actO = neuron.LIFNode(surrogate_function=surrogate.ATan())

        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x):
        out = x
        out = self.conv1(out)
        c1out = out
        out = self.act1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        c2out = out
        out = self.act2(out)
        out = self.maxpool2(out)
        out = self.fcP(out).squeeze(-1)
        outO = self.fcO(out)
        outO = self.output_actO(outO)
        frO = outO.mean(0)

        # spike freq, conv 1 output, conv 2 output
        return frO, c1out, c2out
