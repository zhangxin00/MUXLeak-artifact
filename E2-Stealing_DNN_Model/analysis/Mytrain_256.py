import torch 
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import argparse
import pickle
import Levenshtein
import tqdm
import os, time
from copy import deepcopy
import collections
import multiprocessing as mp
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from dataset import RaplLoader
from config import vocab_size, sos_id, eos_id
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

device_ids = [0]
torch.manual_seed(7)
np.random.seed(7)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def DatasetPreprocess(dataset):
    print(dataset[0][0],len(dataset[0][0]))
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
        input_lengths.append(1000)
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

def train(model, train_dataset, optimizer, scheduler, epoch):
    model.train()
    data_len = len(train_dataset[0])
    losses = AverageMeter()

    shuffled_index=torch.randperm(data_len)
    num_batch, test_loss, oer = 0, 0, 0

    for count in range(0, data_len, args.batch_size):
        train_sequence, train_label, train_sequence_lengths, train_label_lengths = \
        BatchDataProcess(train_dataset, shuffled_index, count, args.batch_size)

        train_sequence.requires_grad_()
        # print("train_sequence:",train_sequence.size())
        # print("train_sequence_length:", train_sequence_lengths.size())
        # print("train_label:",train_label.size())

        pred, gold = model(train_sequence, train_sequence_lengths, train_label)
        loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        losses.update(loss.item())
        if count % 1000 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, count, data_len, loss=losses))

    return losses.avg

def OER(gt_list, hyp_list, truth_len):
    #truth_len = truth_len.tolist()
    batch_size = len(gt_list)
    edit_distance = []
    for i in range(batch_size):
        edit_distance.append(Levenshtein.distance(gt_list[i], hyp_list[i]))
    # batch_oer = list(map(lambda x: x[0]/x[1], zip(edit_distance, truth_len)))
    if(len(hyp_list[0])>0):
        batch_oer = edit_distance[0]/ len(hyp_list[0])
    else:
        batch_oer = 1
    return batch_oer

def test(model, test_dataset, epoch):
    model.eval()
    data_len = len(test_dataset[0])
    losses = AverageMeter()
    index_list = range(0, data_len)
    char_list = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'a', 11: '<sos>',
                 12: '<eos>'}
    total_oer = 0
    '''
    test_sequence, test_label, test_sequence_lengths, test_label_lengths = test_dataset
    for count in range(0, data_len):
        input_seq = test_sequence[count].to(device)
        input_length = test_sequence_lengths[count].unsqueeze(0).to(device)
        label_length = test_label_lengths[count].to(device)
        output_label = test_label[count][:label_length].tolist()
        hyp_list, gt_list = [], []
        with torch.no_grad():
            # Forward prop.
            nbest_hyps = model.recognize(input_seq, input_length, char_list, args)
        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [char_list[idx] for idx in out if idx not in (sos_id, eos_id)]
            out = ''.join(out)
            hyp_list.append(out)

        print(hyp_list)

        gt = [char_list[idx] for idx in output_label if idx not in (sos_id, eos_id)]
        gt = ''.join(gt)
        gt_list.append(gt)
        print(gt_list)

        batch_oer = OER(gt_list, hyp_list, len(hyp_list))
        total_oer += batch_oer
    '''
    for count in range(0, data_len, args.batch_size):
        test_sequence, test_label, test_sequence_lengths, test_label_lengths = \
            BatchDataProcess(test_dataset, index_list, count, args.batch_size)
        hyp_list, gt_list = [], []
        with torch.no_grad():
            # Forward prop.
            pred, gold = model(test_sequence, test_sequence_lengths, test_label)
            loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)
        
        # Keep track of metrics
        losses.update(loss.item())

    print('\nTest Loss {:.6f} ({:.6f})\n'.format(losses.val, losses.avg))
    print('\nTotal OER {:.6f}\n'.format(total_oer))
    avg_oer = total_oer / data_len
    return losses.val



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers_enc', type=int, default=1, help='number of encoder layers.')
    parser.add_argument('--n_layers_dec', type=int, default=1, help='number of decoder layers.')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model input.')
    parser.add_argument('--n_head', type=int, default=8, help='number of heads in MHA.')
    parser.add_argument('--d_k', type=int, default=32, help='Dimension of key')
    parser.add_argument('--d_v', type=int, default=32, help='Dimension of value')
    parser.add_argument('--d_inner', type=int, default=256, help='dimension of feedforward layer.')
    parser.add_argument('--d_word_vec', type=int, default=256, help='Dim of decoder embedding.')
    parser.add_argument('--d_input', type=int, default=96, help='number of input features.')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers to generate minibatch')
    parser.add_argument('--tgt_emb_prj_weight_sharing', default=1, type=int, help='share decoder embedding with decoder projection')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=int, default=5e-4) # 5e-4
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help='label smoothing')
    parser.add_argument('--checkpoint', type=str, default="BasicModel_3600_256.pt")
    parser.add_argument('--pe_maxlen', default=1200, type=int, help='Positional Encoding max len')
    parser.add_argument('--beam_size', default=10, type=int, help='Beam size')
    parser.add_argument('--nbest', default=1, type=int, help='Nbest size')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
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
    train0 = RaplLoader(batch_size=args.batch_size, num_workers=args.workers,mypath='/home/jx/deeppower/mercury-dataset/finaltrain3.h5')
    train_loader,train_dataset = train0.get_loader()
    test0 = RaplLoader(batch_size=args.batch_size, num_workers=args.workers,mypath='/home/jx/deeppower/mercury-dataset/finaltest1.h5')
    test_loader,test_dataset= test0.get_loader()
    print(train_dataset[0])
    print(len(train_loader))

    train_dataset = DatasetPreprocess(train_dataset)
    test_dataset = DatasetPreprocess(test_dataset)
    # print(train_dataset[0].size(), train_dataset[1][0])
    # os._exit(0)

    if (1==2):
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
    else:
        encoder = Encoder(args.d_input, args.n_layers_enc, args.n_head,
                            args.d_k, args.d_v, args.d_model, args.d_inner,
                            dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, vocab_size,
                            args.d_word_vec, args.n_layers_dec, args.n_head,
                            args.d_k, args.d_v, args.d_model, args.d_inner,
                            dropout=args.dropout,
                            tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                            pe_maxlen=args.pe_maxlen)
        model = Transformer(encoder, decoder).to(device)
        #model = torch.nn.DataParallel(model, device_ids=device_ids)
        # model = nn.DataParallel(model, device_ids=[0,1,2,3])

        # optimizer = TransformerOptimizer(
        #         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09), d_model = args.d_model)
        # scheduler = 0

        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=5e-08)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, 
                                                steps_per_epoch=int(len(train_dataset[0])/args.batch_size),
                                                epochs=args.epochs,
                                                anneal_strategy='linear')
        

    # print(model)
    print("Number of model parameters:", sum([param.nelement() for param in model.parameters()]))                          

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        train_loss = train(model, train_dataset, optimizer, scheduler, epoch)
        end_time = time.time()
        print("training for epoch {} takes {} seconds".format(epoch, end_time-start_time))
        # if epoch % 10 == 0 or epoch == args.epochs - 1 :
        start_time = time.time()
        test_loss = test(model, test_dataset, epoch)
        end_time = time.time()
        print("testing for epoch {} takes {} seconds".format(epoch, end_time-start_time))

        # Check if there was an improvement
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0


        # Save checkpoint
        best_path = "BestModel_3600_256.pt"
        basic_path = "BasicModel_3600_256.pt"
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, scheduler, best_loss, is_best, basic_path, best_path)


if __name__ == "__main__":
    main()








