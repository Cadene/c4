from __future__ import print_function
import os
from os import path as osp
import math
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch

def tensorize_board(game_plot):
    board = torch.zeros(6,7)
    hist = torch.zeros(7,dtype=torch.int64)
    bin = 1
    game_plot = str(game_plot)
    for i in range(len(game_plot)):
        pos = int(game_plot[i])
        board[hist[pos-1],pos-1] = bin
        hist[pos-1] = hist[pos-1] + 1
        bin = - bin

    if(hist.sum().abs()%2==1):
        board = -board
    return board


class C4(torch.utils.data.Dataset):

    def __init__(self,
            dir_data='data',
            split='train',
            pc_train=1.,
            seed=1337):
        self.dir_data = dir_data
        self.path_c4dat = osp.join(dir_data, 'c4.dat')
        self.split = split
        self.pc_train = pc_train
        self.seed = seed

        if not osp.isfile(self.path_c4dat):
            self.download_data()
        self.data, self.targets = self.load_data()
        self.classes = ['player1win', 'player2win', 'null']

        self.ids_by_class = [
            [i for i,cid in enumerate(self.targets) if cid > 0],
            [i for i,cid in enumerate(self.targets) if cid < 0],
            [i for i,cid in enumerate(self.targets) if cid == 0]]

        if self.split in ['train', 'val']:
            self.ids = self.random_split_by_class()
        elif self.split == 'trainval':
            self.ids = list(range(len(self.data)))
        else:
            raise ValueError(self.split)

        self.ids_by_class_split = [
            [idx for idx in self.ids if self.targets[idx] > 0],
            [idx for idx in self.ids if self.targets[idx] < 0],
            [idx for idx in self.ids if self.targets[idx] == 0]]

        print(f'{self.split}set_length_cid,0', len(self.ids_by_class_split[0]))
        print(f'{self.split}set_length_cid,1', len(self.ids_by_class_split[1]))
        print(f'{self.split}set_length_cid,2', len(self.ids_by_class_split[2]))
        print(f'{self.split}set_pc_cid,0',
            len(self.ids_by_class_split[0])/len(self.ids_by_class[0])*100)
        print(f'{self.split}set_pc_cid,1',
            len(self.ids_by_class_split[1])/len(self.ids_by_class[1])*100)
        print(f'{self.split}set_pc_cid,2',
            len(self.ids_by_class_split[2])/len(self.ids_by_class[2])*100)
        print(f'{self.split}set_length', len(self))

    def download_data(self):
        os.system('mkdir -p ' + self.dir_data)
        os.system(f'wget -O {self.path_c4dat} http://data.lip6.fr/cadene/rl/c4.dat')

    def load_data(self):
        raw = np.loadtxt(self.path_c4dat, dtype=np.int64)
        d = raw[:,0].squeeze()
        t = raw[:,2].squeeze()
        return d, t

    def random_split_by_class(self):
        ids_by_class = [
            self.random_split(len(self.ids_by_class[0]), seed=self.seed),
            self.random_split(len(self.ids_by_class[1]), seed=self.seed+1),
            self.random_split(len(self.ids_by_class[2]), seed=self.seed+2)]
        ids = []
        ids += [self.ids_by_class[0][idx] for idx in ids_by_class[0]]
        ids += [self.ids_by_class[1][idx] for idx in ids_by_class[1]]
        ids += [self.ids_by_class[2][idx] for idx in ids_by_class[2]]
        return ids

    def random_split(self, length, seed=None):
        rnd = np.random.RandomState(seed=seed)
        indices = rnd.choice(length,
                             size=int(length*self.pc_train),
                             replace=False)
        if self.split == 'val':
            indices = np.array(list(set(np.arange(length)) - set(indices)))
        return indices.tolist()

    def __getitem__(self, index):
        idx = self.ids[index]
        game_plot, target = self.data[idx], self.targets[idx]

        if target > 0:
            target = 0 # player1win
        elif target < 0:
            target = 1 # player2win
        elif target == 0:
            target = 2 # null
        else:
            raise ValueError(target)
        target = torch.LongTensor([target])

        board = tensorize_board(game_plot)
        board = board.unsqueeze(0)
        target = target.item()
        return board, target

    def __len__(self):
        return len(self.ids)

    def make_batch_loader(self):
        data_loader = data.DataLoader(self,
            batch_size=self.batch_size,
            num_workers=self.nb_threads,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=False)
        return data_loader
