import torch 
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from dataset import RaplLoader
import torch.optim as optim
import numpy as np
import argparse
import pickle
import Levenshtein
import tqdm
import os, time
from copy import deepcopy

from config import vocab_size, sos_id, eos_id
from data_gen import pad_collate
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger
from xer import cer_function

torch.manual_seed(7)
np.random.seed(7)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def DatasetPreprocess(dataset):
    #print(dataset[0])
    input_sequences = dataset[:][0]
    labels = dataset[:][1]
    #new_sequences = torch.Tensor(input_sequences).unsqueeze(dim=2)
    new_sequences = torch.Tensor(input_sequences).reshape(len(dataset[:][0]), 1200, 3)
    #print(new_sequences.size())
    new_labels = []
    input_lengths=[]
    label_lengths=[]

    for i, label in enumerate(labels):
        label_lengths.append(len(label))
        new_labels.append(label[:label_lengths[i]])
        input_lengths.append(1200)
  #  new_labels=torch.Tensor(new_labels)
    new_tensor = torch.full((len(input_lengths), 16), -1)
    for i, arr in enumerate(new_labels):
        new_tensor[i, :len(arr)] = torch.tensor(arr, dtype=torch.int32)
    #print(new_labels)
   # new_labels = nn.utils.rnn.pad_sequence(torch.Tensor(new_labels), batch_first=True, padding_value = -1)

    new_dataset = [new_sequences, new_tensor, torch.tensor(input_lengths), torch.tensor(label_lengths)]

    return new_dataset


def BatchDataProcess(dataset, index_list, count, batch_size):
    input_sequences, labels, input_lengths, label_lengths = dataset
    index = index_list[count : count + batch_size]
    batch_sequence=input_sequences[index].to(device)
    batch_sequence.requires_grad_()
    batch_label=labels[index].to(device=device, dtype=torch.int64)
    batch_sequence_lengths = input_lengths[index].to(device)
    batch_label_lengths = label_lengths[index].to(device)

    return batch_sequence, batch_label, batch_sequence_lengths, batch_label_lengths

def OER(gt_list, hyp_list, truth_len):
    #truth_len = truth_len.tolist()
    batch_size = len(gt_list)
    edit_distance = []
    for i in range(batch_size):
        edit_distance.append(Levenshtein.distance(gt_list[i], hyp_list[i]))
    # batch_oer = list(map(lambda x: x[0]/x[1], zip(edit_distance, truth_len)))
    batch_oer = edit_distance[0]/ (len(gt_list[0])+2)
    return batch_oer

bad_case=['3312','313312','31333312','331331333312']
case={}
count_case = {}
acc1=0
acc2=0
acc3=0
cnt=0
def test(model, test_dataset):
    global cnt,acc1,acc2,acc3
    model.eval()
    data_len = len(test_dataset[0])
    losses = AverageMeter()
    index_list = range(0, data_len)
    char_list = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'a', 11:'<sos>', 12:'<eos>'}
    total_oer = 0
    model_oer=[]
    moer=0
    bad=[]

    test_sequence, test_label, test_sequence_lengths, test_label_lengths = test_dataset
    for count in range(0, data_len):
        label_length = test_label_lengths[count].to(device)
        output_label = test_label[count][:label_length].tolist()
        gt = [char_list[idx] for idx in output_label if idx not in (sos_id, eos_id)]
        gt = ''.join(gt)
        if gt in count_case:
            count_case[gt] += 1
        else:
            count_case[gt] = 1
        case[gt]=0
    print(case)
    for count in range(0, data_len):

        input_seq = test_sequence[count].to(device)
        input_length = test_sequence_lengths[count].unsqueeze(0).to(device)
        label_length = test_label_lengths[count].to(device)
        output_label = test_label[count][:label_length].tolist()
                
        # print("phase 1 start!")
        # start_time = time.time()       
        with torch.no_grad():
           nbest_hyps = model.recognize(input_seq, input_length, char_list, args)              
        # end_time = time.time()
        # print("phase 1 finish! time is {} seconds".format(end_time - start_time))

        hyp_list, gt_list = [], []

        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [char_list[idx] for idx in out if idx not in (sos_id, eos_id)]
            out = ''.join(out)
            hyp_list.append(out)

        print("predict", hyp_list)

        gt = [char_list[idx] for idx in output_label if idx not in (sos_id, eos_id)]
        gt = ''.join(gt)
        gt_list.append(gt)
        print("truth", gt_list)
        cnt+=1
        acc1+=(float) (abs(hyp_list[0].count("1")-gt_list[0].count("1")))/hyp_list[0].count("1")
        acc2 += (float)(abs(hyp_list[0].count("2") - gt_list[0].count("2")) )/ hyp_list[0].count("2")
        acc3 += (float)(abs(hyp_list[0].count("3") - gt_list[0].count("3"))) / hyp_list[0].count("3")

        batch_oer = OER(gt_list, hyp_list, len(hyp_list)+2)
        total_oer += batch_oer
        case[gt]=case[gt]+batch_oer/count_case[gt]



        print("Batch: {}, aver_batch_oer:{:6f}".format(count, batch_oer))

    f=zip(case.values(),case.keys())
    ds=sorted(f)
    print(ds)
    myoer=0
    mycnt=0
    for v,c in ds:
        if mycnt<10:
            myoer=myoer+v
            mycnt+=1

    avg_oer = total_oer / data_len
    print('Avg_oer {:.6f}\n'.format(avg_oer))
    print('Best_oer {:.6f}\n'.format(myoer/10))
    print("pool:", 1 - acc1 / cnt, " full connect:", 1 - acc2 / cnt, " conv:", 1 - acc3 / cnt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers_enc', type=int, default=1, help='number of encoder layers.')
    parser.add_argument('--n_layers_dec', type=int, default=1, help='number of decoder layers.')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model imput.') 
    parser.add_argument('--n_head', type=int, default=8, help='number of heads in MHA.')
    parser.add_argument('--d_k', type=int, default=32, help='Dimension of key')
    parser.add_argument('--d_v', type=int, default=32, help='Dimension of value')
    parser.add_argument('--d_inner', type=int, default=128, help='dimension of feedforward layer.')
    parser.add_argument('--d_word_vec', type=int, default=256, help='Dim of decoder embedding.')
    parser.add_argument('--d_input', type=int, default=96, help='number of input features.')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers to generate minibatch')
    parser.add_argument('--tgt_emb_prj_weight_sharing', default=1, type=int, help='share decoder embedding with decoder projection')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=int, default=0.001) # 5e-4
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help='label smoothing')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pe_maxlen', default=1200, type=int, help='Positional Encoding max len')
    parser.add_argument('--beam_size', default=1, type=int, help='Beam size')
    parser.add_argument('--nbest', default=1, type=int, help='Nbest size')
    parser.add_argument('--decode_max_len', default=0, type=int,
                        help='Max output length. If ==0 (default), it uses a '
                             'end-detect function to automatically find maximum '
                             'hypothesis lengths')
    global args
    args = parser.parse_args()

    start_epoch = 0
    best_loss = float('inf')
    epochs_since_improvement = 0

    ## load data and label
    ## train/test dataset format: [[[input_features]], [[label]], [input_length], [label_length]], each list has number_of_sequences elements
    print('Loading data...')
    train0 = RaplLoader(batch_size=args.batch_size, num_workers=2,mypath='/home/jx/deeppower/mercury-dataset/mercurytrain3.h5')
    train_loader,train_dataset = train0.get_loader()
    test0 = RaplLoader(batch_size=args.batch_size, num_workers=2,mypath='/home/jx/deeppower/mercury-dataset/finaltest1.h5')
    test_loader,test_dataset= test0.get_loader()

    train_dataset = DatasetPreprocess(train_dataset)
    test_dataset = DatasetPreprocess(test_dataset)                   

    checkpoint = "BasicModel_3600_256.pt"
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model = checkpoint['model'].to(device)

    start_time = time.time()
    test(model, test_dataset)
    end_time = time.time()
    print("testing takes {} seconds".format(end_time-start_time))



if __name__ == "__main__":
    main()








