import os
import wfdb
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from resnet1d.resnet1d import ResNet1D


def load_physio_data(path):
    data_list = []
    label_list = []
    dir_list = os.listdir(path)
    for f in tqdm(dir_list, desc='Processing'):
        if f.find('hea') == -1:
            continue
        f = os.path.splitext(f)[0]
        file = os.path.join(path, f)
        data = wfdb.rdrecord(file)
        label = f.split('_')[2]
        if label == 'B':
            label = "VB"
        label_list.append(label)
        data_list.append(data.p_signal.squeeze(1))

    data_list = np.array(data_list)
    data_list = pd.DataFrame(data_list)
    label_list = pd.Series(label_list, name='label')

    return pd.concat([data_list, label_list], axis=1)


def load_ResNet1D(ord=True, path=None):
    model = ResNet1D(
        in_channels=1,
        base_filters=32,
        kernel_size=31,
        stride=2,
        groups=1,
        n_block=5,
        n_classes=4 - int(ord),
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=True,
        verbose=False)

    return model


if __name__ == '__main__':
    load_physio_data('datasets/FRAG/all_frag').to_pickle('fragment.pkl')
    # load_all_data()
