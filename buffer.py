import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Buffer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args   = args
        buffer_size = args.mem_size 
        if args.gen: 
            buffer_size = (args.mem_size * 48) 
            print('number of (integers) stored : %d' % (buffer_size * 8 * 8))
            print('number of real images that could be stored : %d' % (buffer_size // 48))

        print('buffer has %d slots' % buffer_size)

        # for reservoir only (nb: not actually used)
        self.min_per_class = args.mem_size * args.n_tasks 

        # for VQ-VAE, we save indices, not actual floats
        if args.gen : 
            bx = torch.LongTensor(buffer_size, *args.input_size).to(args.device).fill_(0)
        else:
            bx = torch.FloatTensor(buffer_size, *args.input_size).to(args.device).fill_(0)

        by = torch.LongTensor(buffer_size).to(args.device).fill_(0)
        bt = torch.LongTensor(buffer_size).to(args.device).fill_(0)
        logits = torch.FloatTensor(buffer_size, args.n_classes).to(args.device).fill_(0)

        self.current_index = 0
        self.n_seen_so_far = 0

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('bt', bt)
        self.register_buffer('logits', logits)

        self.to_one_hot  = lambda x : x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]


    @property
    def x(self):
        return self.bx[:self.current_index]

    @property 
    def y(self):
        return self.to_one_hot(self.by[:self.current_index])

    @property
    def t(self):
        return self.bt[:self.current_index]

    def display(self):
        from torchvision.utils import save_image
        from PIL import Image

        if 'cifar' in self.args.dataset:
            shp = (-1, 3, 32, 32)
        else:
            shp = (-1, 1, 28, 28)

        save_image((self.x.reshape(shp) * 0.5 + 0.5), 'tmp.png', nrow=int(self.current_index ** 0.5))
        Image.open('tmp.png').show()
        print(self.by[:self.current_index])

    def add_reservoir(self, x, y, logits, t):
        n_elem = x.size(0)
        save_logits = logits is not None

        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.bt[self.current_index: self.current_index + offset].fill_(t)
   
            if save_logits: 
                self.logits[self.current_index: self.current_index + offset].data.copy_(logits[:offset])
        
            self.current_index += offset
            self.n_seen_so_far += offset
            
            # everything was added
            if offset == x.size(0): 
                return 

        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]

        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices = (indices < self.bx.size(0)).long() 

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        # perform overwrite op
        self.bx[idx_buffer] = x[idx_new_data]
        self.by[idx_buffer] = y[idx_new_data]
        self.bt[idx_buffer] = t

        if save_logits: 
            self.logits[idx_buffer] = logits[idx_new_data]

        self.n_seen_so_far += x.size(0)
       
    def sample(self, amt, exclude_task=None):
        if exclude_task is not None:
            valid_indices = (self.t != exclude_task).nonzero().squeeze()
            bx, by = self.bx[valid_indices], self.by[valid_indices]
        else:
            bx, by = self.bx[:self.current_index], self.by[:self.current_index]

        if bx.size(0) < amt:
            # return self.bx[:self.current_index], self.by[:self.current_index]
            return bx, by
        else:
            # indices = torch.from_numpy(np.random.choice(self.current_index, amt, replace=False)).to(self.args.device)
            indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False)).to(self.args.device)
            return bx[indices], by[indices]

    def split(self, amt):
        indices = torch.randperm(self.current_index).to(self.args.device)
        return indices[:amt], indices[amt:]


def get_cifar_buffer(args):
    args.input_size = (8, 8)

    return Buffer(args)


if __name__ == '__main__':
    class args:
        pass
    args.n_tasks = 10
    args.bufffer_samples_per_task = 5
    args.n_classes = 10
    args.device = 'cuda:0'
    args.input_size = 784

    buffer = Buffer(args)

    x = torch.randn(10, 784).to(args.device)
    y = torch.randn(10).uniform_(0, args.n_classes).long().to(args.device)

    for _ in range(10):
        buffer.maybe_add(x,y, 0)

        

