import model
import paddle
import numpy as np
import argparse
from reprod_log import ReprodLogger
import argparse
import os
import pickle
import random
import numpy as np
import csv

reprod_logger = ReprodLogger()

parser = argparse.ArgumentParser(description='Paddle Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'CNN_MLP'], default='RN',
                    help='resume from model stored')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=True,
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
    
    print("accuracy_ternary",accuracy_ternary)

    accuracy_ternary = sum(accuracy_ternary) / len(accuracy_ternary)
    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    print('\n Test set: Ternary accuracy: {:.0f}% Binary accuracy: {:.0f}% | Unary accuracy: {:.0f}%\n'.format(
        accuracy_ternary, accuracy_rel, accuracy_norel))

    loss_ternary = sum(loss_ternary) / len(loss_ternary)
    loss_binary = sum(loss_binary) / len(loss_binary)
    loss_unary = sum(loss_unary) / len(loss_unary)

    return accuracy_ternary, accuracy_rel, accuracy_norel



model = model.RN(args)
print(model)

weight_dict = paddle.load('epoch_RN_25.pdparams')
model.set_state_dict(weight_dict)


print('loading data...')
dirs = './data'
filename = os.path.join(dirs,'sort-of-clevr.pickle')
with open(filename, 'rb') as f:
    train_datasets, test_datasets = pickle.load(f)

ternary_test = []
rel_test = []
norel_test = []
print('processing data...')

count = 0

for img, ternary, relations, norelations in test_datasets:

    count = count+1
    
    img = np.swapaxes(img, 0, 2)
    for qst, ans in zip(ternary[0], ternary[1]):
        ternary_test.append((img, qst, ans))
    for qst,ans in zip(relations[0], relations[1]):
        rel_test.append((img,qst,ans))
    for qst,ans in zip(norelations[0], norelations[1]):
        norel_test.append((img,qst,ans))

    if count == 64:
        test_acc_ternary, test_acc_binary, test_acc_unary = test(1, ternary_test, rel_test, norel_test)

        print(len(rel_test))
        print(len(norel_test))



