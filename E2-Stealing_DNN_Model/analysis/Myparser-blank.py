import numpy as np
import json
import sys
import os
import warnings

'''
0 --- conv
1 --- pooling
2 --- fc
'''


# print(path)
def load(path):
    save_inputs = []
    save_targets = []
    save_position = []
    if os.path.isdir(path):
        # If directory, find all .pkl files
        filepaths = [os.path.join(path, x) for x in os.listdir(path) if x.find('model')]
    elif os.path.isfile(path):
        # If single file, just use this one
        filepaths = [path]
    cnt = 0
    t = 100
    k = 0.005883
    ii = 0
    ff = []
    maxl = 0
    for file in filepaths:
        tname = file + '/trace.txt'
        lname = file + '/layer.txt'
        print(tname, lname)
        count = []
        cnt = 0
        begin = []
        end = []

        position = []
        targets = []

        l = 0

        f = open(tname, mode='r')
        for line in f:
            l += int(line)
            cnt += 1
            if (cnt == t):
                count.append(l)
                l = 0
                cnt = 0
        bf = 0
        print(len(count))
        pos_start = 0
        with open(lname, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    values = line.split(',')
                    A = values[0].strip()
                    B = int(values[1].strip())
                    E = int(values[2].strip())
                    if (len(end) == 0):
                        add2 = int((E - B) * k)
                        add1 = 0
                    else:
                        add2 = int((E - end[len(end) - 1]) * k)

                    # position.append([len(targets), len(targets) + add-1])
                    if (A[1] == 't'):
                        pad_start = add2 - 1
                        # position.append(0)
                        # targets = np.pad(targets, (0, add2-1), 'constant', constant_values=0)
                        continue
                    elif (len(end) != 0):
                        add1 = int((B - end[len(end) - 1]) * k)
                        if (add1 < 2):
                            add1 = 2
                        if (add1 > 5):
                            add1 = add1 - add1 // 5
                        # targets.append(0)
                        # targets = np.pad(targets, (0, add1-1), 'constant', constant_values=0)
                    if (A[1] == 'p'):
                        if (add2 > 5):
                            targets.append(1)
                            # targets = np.pad(targets, (0, add2-add2//20), 'constant', constant_values=1)
                        else:
                            targets.append(1)
                            # targets = np.pad(targets, (0, add2-1), 'constant', constant_values=1)
                    elif (A[1] == 'f'):
                        if (add2 > 5):
                            targets.append(2)
                            # targets = np.pad(targets, (0, add2 - add2//20), 'constant', constant_values=2)
                        else:
                            targets.append(2)
                            # targets = np.pad(targets, (0, add2), 'constant', constant_values=2)
                    else:
                        if (add2 > 18):
                            targets.append(3)
                            # targets = np.pad(targets, (0, add2-add2//20), 'constant', constant_values=3)
                        else:
                            targets.append(3)
                            # targets = np.pad(targets, (0, add2-1), 'constant', constant_values=3)

                    begin.append(B)
                    end.append(E)

        # print("before:",len(count))
        if (len(count) > maxl):
            maxl = len(count)
        print("maxl:", maxl)
        # count=count[pad_start:]
        # print(len(count))
        print(file, len(position), len(count), len(targets), pos_start)

        # 填充count
        count = np.pad(count, (0, 3600 - len(count)), 'constant', constant_values=0)

        if (len(targets) > 3):
            save_position.append(position)
            save_targets.append(targets)
            save_inputs.append(count)
            ff.append(file)
        # print(position[1],position[1][1])
    return save_inputs, save_targets, ff


r1, r2, ff = load('../FPGA-dataset/finaltrain3')
data = []
print(ff)
# for i in range(len(r2)):
#     temp = []
#     for j in range(len(r2[i])):
#         temp.append(r1[i][j])
#     data.append(temp)
import h5py

f = h5py.File('./finaltrain3.h5', 'w')
g1 = f.create_group('data')
g2 = f.create_group('targets')

for i in range(len(r1)):
    # print(i)
    # print(ff[i])
    g1.create_dataset("custom" + str(i), data=r1[i])
for i in range(len(r2)):
    print(r2[i])
    g2.create_dataset("custom" + str(i), data=r2[i])
# f['data']=save_inputs

f.close
