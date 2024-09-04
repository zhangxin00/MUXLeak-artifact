import pickle
import random

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from config import pickle_file, IGNORE_ID


def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')

    for elem in batch:
        feature, trn = elem
        max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
        max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

    for i, elem in enumerate(batch):
        feature, trn = elem
        input_length = feature.shape[0]
        input_dim = feature.shape[1]
        padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
        padded_input[:input_length, :] = feature
        padded_target = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=IGNORE_ID)
        batch[i] = (padded_input, padded_target, input_length)

    # sort it by input lengths (long to short)
    batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)