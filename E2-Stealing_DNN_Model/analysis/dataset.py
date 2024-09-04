import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import h5py
import os
from sklearn.model_selection import train_test_split


labels_name = { 'bound': 0, 'pooling': 1,
               'fc': 2,'conv':3,'_': 4}
'''
0 --- start
1 --- conv
2 --- pooling
3 --- fc
'''

class Rapl(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.x,self.y = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        #print(self.x.size(),self.y.size())
        x, y = self.x[index], self.y[index]
        feature,  label = x, y
        #feature = self.transform(feature) if self.transform is not None else feature
        #feature = feature.resize(3,1000)
        #target = self.target_transform(target) if self.target_transform is not None else target
        return feature, label

    def __len__(self):
        return len(self.x)


def collate_fn_batch(data):
    #max_length1 = max([_x.shape[0] for (_x, _y) in data])
    #max_length1 += 16 - max_length1 % 16
    max_length2 = max([_y.shape[0] for (_x, _y) in data])
    max_length2 += 16 - max_length2 % 16
    #print(max_length1,max_length2)
    x, y, z = [], [], []
    for i, (_x, _y) in enumerate(data):
        #print(i)
        #print(_x)
        #print(_y)
        _x, _y = torch.as_tensor(_x), torch.as_tensor(_y)
        #print(_x.size(),_y.size())
        # _x, _y = torch.as_tensor(_x).transpose(1, 0), torch.as_tensor(_y).transpose(1, 0)
        # l = int((max_length - _x.shape[-1]) / 2)
        l = 0
        #r1 = max_length1 - _x.shape[-1] - l
        r2 = max_length2 - _y.shape[-1] - l
        #_x = F.pad(_x, (l, r1), "constant", 0)
        _y = F.pad(_y, (l, r2), "constant", 0)
        #print("r2:",r2)

        x = _x.unsqueeze(0) if i == 0 else torch.concat([x, _x.unsqueeze(0)])
        #print(x.size())
        #x = x.resize(3,1000)
        #print(x.size())
        y = _y.unsqueeze(0) if i == 0 else torch.concat([y, _y.unsqueeze(0)])
        #y = _y if i == 0 else torch.concat([y, _y])
    #print(x.size(), y.size())

    return x, y


class RaplLoader(object):
    def __init__(self, batch_size, num_workers=0, mypath=""):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = labels_name['_']
        self.path = mypath
        self.data = self.preprocess()
        # print(self.num_classes)

    def preprocess(self):
        x, y = [], []
        #print(self.path)
        data = h5py.File(self.path, 'r')
        #print(data.keys())
        #print(data['data'].keys())
        #print(data['targets'].keys())
        #data = h5py.File(r'../datasets/data.h5', 'r')
        for k in data['data'].keys():
            x.append(data['data'][k][:])
            y.append(data['targets'][k][:])

        return x, y

    def loader(self, data, shuffle=False, transform=None, target_transform=None):
        dataset = Rapl(data, transform=transform, target_transform=target_transform)
        #print(dataset)
        #print(data)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, collate_fn=collate_fn_batch)
        return dataloader,dataset

    def get_loader(self):
        #print(self.data)
        dataloader = self.loader(self.data, shuffle=True)
        return dataloader
