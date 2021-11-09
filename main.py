from __future__ import print_function
import argparse
import os
import pickle
import random
import numpy as np
import csv


import paddle
from model import RN, CNN_MLP


# Training settings
parser = argparse.ArgumentParser(description='Paddle Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN', 
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--relation-type', type=str, default='binary',
                    help='what kind of relations to learn. options: binary, ternary (default: binary)')

args = parser.parse_args()

paddle.seed(args.seed)

if args.model=='CNN_MLP': 
  model = CNN_MLP(args)
else:
  model = RN(args)
  
model_dirs = './model'
bs = args.batch_size
input_img = paddle.empty(shape=[bs, 3, 75, 75])
input_qst = paddle.empty(shape=[bs, 18])
label = paddle.empty(shape=[bs],dtype='int64')

def tensor_data(data, i):
    img = paddle.to_tensor(np.asarray(data[0][bs*i:bs*(i+1)]),dtype='float32')
    qst = paddle.to_tensor(np.asarray(data[1][bs*i:bs*(i+1)]),dtype='float32')
    ans = paddle.to_tensor(np.asarray(data[2][bs*i:bs*(i+1)]))

    global input_img
    global input_qst
    global label
    input_img = img
    input_qst = qst
    label = ans

def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

    
def train(epoch, ternary, rel, norel):
    model.train()

    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    random.shuffle(ternary)
    random.shuffle(rel)
    random.shuffle(norel)

    ternary = cvt_data_axis(ternary)
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    acc_ternary = []
    acc_rels = []
    acc_norels = []

    l_ternary = []
    l_binary = []
    l_unary = []

    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(ternary, batch_idx)
        accuracy_ternary, loss_ternary = model.train_(input_img, input_qst, label)
        acc_ternary.append(accuracy_ternary)
        l_ternary.append(loss_ternary)

        tensor_data(rel, batch_idx)
        accuracy_rel, loss_binary = model.train_(input_img, input_qst, label)
        acc_rels.append(accuracy_rel)
        l_binary.append(loss_binary)

        tensor_data(norel, batch_idx)
        accuracy_norel, loss_unary = model.train_(input_img, input_qst, label)
        acc_norels.append(accuracy_norel)
        l_unary.append(loss_unary)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] '
                  'Ternary accuracy: {:.0f}% | Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(
                   epoch,
                   batch_idx * bs * 2,
                   len(rel[0]) * 2,
                   100. * batch_idx * bs / len(rel[0]),
                   accuracy_ternary,
                   accuracy_rel,
                   accuracy_norel))
        
    avg_acc_ternary = sum(acc_ternary) / len(acc_ternary)
    avg_acc_binary = sum(acc_rels) / len(acc_rels)
    avg_acc_unary = sum(acc_norels) / len(acc_norels)


    avg_loss_ternary = sum(l_ternary) / len(l_ternary)
    avg_loss_binary = sum(l_binary) / len(l_binary)
    avg_loss_unary = sum(l_unary) / len(l_unary)


    return avg_acc_ternary, avg_acc_binary, avg_acc_unary

def test(epoch, ternary, rel, norel):
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    ternary = cvt_data_axis(ternary)
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    accuracy_ternary = []
    accuracy_rels = []
    accuracy_norels = []

    loss_ternary = []
    loss_binary = []
    loss_unary = []

    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(ternary, batch_idx)
        acc_ter, l_ter = model.test_(input_img, input_qst, label)
        accuracy_ternary.append(acc_ter)
        loss_ternary.append(l_ter)

        tensor_data(rel, batch_idx)
        acc_bin, l_bin = model.test_(input_img, input_qst, label)
        accuracy_rels.append(acc_bin)
        loss_binary.append(l_bin)

        tensor_data(norel, batch_idx)
        acc_un, l_un = model.test_(input_img, input_qst, label)
        accuracy_norels.append(acc_un)
        loss_unary.append(l_un)

    accuracy_ternary = sum(accuracy_ternary) / len(accuracy_ternary)
    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    print('\n Test set: Ternary accuracy: {:.0f}% Binary accuracy: {:.0f}% | Unary accuracy: {:.0f}%\n'.format(
        accuracy_ternary, accuracy_rel, accuracy_norel))


    loss_ternary = sum(loss_ternary) / len(loss_ternary)
    loss_binary = sum(loss_binary) / len(loss_binary)
    loss_unary = sum(loss_unary) / len(loss_unary)


    return accuracy_ternary, accuracy_rel, accuracy_norel

    
def load_data():
    print('loading data...')
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)
    ternary_train = []
    ternary_test = []
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, ternary, relations, norelations in train_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst, ans in zip(ternary[0], ternary[1]):
            ternary_train.append((img,qst,ans))
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, ternary, relations, norelations in test_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst, ans in zip(ternary[0], ternary[1]):
            ternary_test.append((img, qst, ans))
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))
    
    return (ternary_train, ternary_test, rel_train, rel_test, norel_train, norel_test)
    

ternary_train, ternary_test, rel_train, rel_test, norel_train, norel_test = load_data()

try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

if args.resume:
    filename = os.path.join(model_dirs, args.resume)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = paddle.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

with open(f'./{args.model}_{args.seed}_log.csv', 'w') as log_file:
    csv_writer = csv.writer(log_file, delimiter=',')
    csv_writer.writerow(['epoch', 'train_acc_ternary', 'train_acc_rel',
                     'train_acc_norel', 'train_acc_ternary', 'test_acc_rel', 'test_acc_norel'])

    print(f"Training {args.model} {f'({args.relation_type})' if args.model == 'RN' else ''} model...")

    for epoch in range(1, args.epochs + 1):
        train_acc_ternary, train_acc_binary, train_acc_unary = train(
            epoch, ternary_train, rel_train, norel_train)
        test_acc_ternary, test_acc_binary, test_acc_unary = test(
            epoch, ternary_test, rel_test, norel_test)

        csv_writer.writerow([epoch, train_acc_ternary, train_acc_binary,
                         train_acc_unary, test_acc_ternary, test_acc_binary, test_acc_unary])
        model.save_model(epoch)