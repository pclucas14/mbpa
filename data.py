import torch
import numpy as np

from torchvision import datasets, transforms

""" Template Dataset with Labels """
class XYDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, **kwargs):
        self.x, self.y = x, y

        # this was to store the inverse permutation in permuted_mnist
        # so that we could 'unscramble' samples and plot them
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].float() / 255.
        y = self.y[idx].long()

        # for some reason mnist does better \in [0,1] than [-1, 1]
        if self.source == 'mnist':
            return x, y
        else:
            return (x - 0.5) * 2, y


""" Template Dataset for Continual Learning """
class CLDataLoader(object):
    def __init__(self, datasets_per_task, args, train=True):
        bs = args.batch_size if train else 64
        
        self.datasets = datasets_per_task
        self.loaders = [
                torch.utils.data.DataLoader(x, batch_size=bs, shuffle=True, drop_last=not train) 
                for x in self.datasets ]

    def __getitem__(self, idx):
        return self.loaders[idx]

    def __len__(self):
        return len(self.loaders)


""" Permuted MNIST """
def get_permuted_mnist(args):
    assert not args.use_conv
    
    # fetch MNIST
    train = datasets.MNIST('../cl-pytorch/data/', train=True,  download=True)
    test  = datasets.MNIST('../cl-pytorch/data/', train=False, download=True)

    train_x, train_y = train.train_data, train.train_labels
    test_x,  test_y  = test.test_data,   test.test_labels
    
    train_x = train_x.view(train_x.size(0), -1)
    test_x  = test_x.view(test_x.size(0), -1)

    train_ds, test_ds, inv_perms = [], [], []
    for task in range(args.n_tasks):
        perm = torch.arange(train_x.size(-1)) if task == 0 else torch.randperm(train_x.size(-1))
        
        # build inverse permutations, so we can display samples
        inv_perm = torch.zeros_like(perm)
        for i in range(perm.size(0)):
            inv_perm[perm[i]] = i

        inv_perms += [inv_perm]
        train_ds  += [(train_x[:, perm], train_y)]
        test_ds   += [(test_x[:, perm],  test_y)]
    
    train_ds = map(lambda x, y : XYDataset(x[0], x[1], **{'inv_perm': y, 'source': 'mnist'}), train_ds, inv_perms)
    test_ds  = map(lambda x, y : XYDataset(x[0], x[1], **{'inv_perm': y, 'source': 'mnist'}), test_ds,  inv_perms)

    return train_ds, test_ds        
    

""" Split MNIST into 5 tasks {{0,1}, ... {8,9}} """ 
def get_split_mnist(args):
    assert args.n_tasks in [5, 10], 'SplitMnist only works with 5 or 10 tasks'
    assert '1.' in str(torch.__version__)[:2], 'Use Pytorch 1.x!'

    # fetch MNIST
    train = datasets.MNIST('../cl-pytorch/data/', train=True,  download=True)
    test  = datasets.MNIST('../cl-pytorch/data/', train=False, download=True)

    train_x, train_y = train.train_data, train.train_labels
    test_x,  test_y  = test.test_data,   test.test_labels

    # sort according to the label
    out_train = [
        (x,y) for (x,y) in sorted(zip(train_x, train_y), key=lambda v : v[1]) ]
    
    out_test = [
        (x,y) for (x,y) in sorted(zip(test_x, test_y), key=lambda v : v[1]) ]

    train_x, train_y = [
            torch.stack([elem[i] for elem in out_train]) for i in [0,1] ]
    
    test_x,  test_y  = [
            torch.stack([elem[i] for elem in out_test]) for i in [0,1] ]

    if args.use_conv:
        train_x = train_x.unsqueeze(1)
        test_x  = test_x.unsqueeze(1)
    else:
        train_x = train_x.view(train_x.size(0), -1)
        test_x  = test_x.view(test_x.size(0), -1)

    # get indices of class split
    train_idx = [((train_y + i) % 10).argmax() for i in range(10)]
    train_idx = [0] + sorted(train_idx)

    test_idx  = [((test_y + i) % 10).argmax() for i in range(10)]
    test_idx  = [0] + sorted(test_idx) 

    train_ds, test_ds = [], []
    skip = 10 // args.n_tasks
    for i in range(0, 10, skip):
        tr_s, tr_e = train_idx[i], train_idx[i + skip]
        te_s, te_e = test_idx[i],  test_idx[i + skip]

        train_ds += [(train_x[tr_s:tr_e], train_y[tr_s:tr_e])]
        test_ds  += [(test_x[te_s:te_e],  test_y[te_s:te_e])]

    train_ds = map(lambda x : XYDataset(x[0], x[1]), train_ds)
    test_ds  = map(lambda x : XYDataset(x[0], x[1]), test_ds)

    return train_ds, test_ds

def get_split_cifar(args):
    # assert args.n_tasks in [5, 10], 'SplitCifar only works with 5 or 10 tasks'
    assert '1.' in str(torch.__version__)[:2], 'Use Pytorch 1.x!'

    # fetch MNIST
    train = datasets.CIFAR10('../cl-pytorch/data/', train=True,  download=True)
    test  = datasets.CIFAR10('../cl-pytorch/data/', train=False, download=True)

    train_x, train_y = train.data, train.targets
    test_x,  test_y  = test.data,   test.targets

    # sort according to the label
    out_train = [
        (x,y) for (x,y) in sorted(zip(train_x, train_y), key=lambda v : v[1]) ]
    
    out_test = [
        (x,y) for (x,y) in sorted(zip(test_x, test_y), key=lambda v : v[1]) ]

    train_x, train_y = [
            np.stack([elem[i] for elem in out_train]) for i in [0,1] ]
    
    test_x,  test_y  = [
            np.stack([elem[i] for elem in out_test]) for i in [0,1] ]

    train_x = torch.Tensor(train_x).permute(0, 3, 1, 2).contiguous()
    test_x  = torch.Tensor(test_x).permute(0, 3, 1, 2).contiguous()
    
    train_y = torch.Tensor(train_y)
    test_y  = torch.Tensor(test_y)

    # get indices of class split
    train_idx = [((train_y + i) % 10).argmax() for i in range(10)]
    train_idx = [0] + sorted(train_idx)

    test_idx  = [((test_y + i) % 10).argmax() for i in range(10)]
    test_idx  = [0] + sorted(test_idx) 

    train_ds, test_ds = [], []
    skip = 10 // 5 #args.n_tasks
    for i in range(0, 10, skip):
        tr_s, tr_e = train_idx[i], train_idx[i + skip]
        te_s, te_e = test_idx[i],  test_idx[i + skip]

        train_ds += [(train_x[tr_s:tr_e], train_y[tr_s:tr_e])]
        test_ds  += [(test_x[te_s:te_e],  test_y[te_s:te_e])]

    train_ds = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), train_ds)
    test_ds  = map(lambda x : XYDataset(x[0], x[1], **{'source':'cifar10'}), test_ds)

    return train_ds, test_ds
