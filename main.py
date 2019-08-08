import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import mbpa as MBPA

from data   import *
from buffer import Buffer
from copy   import deepcopy
from pydoc  import locate
from model  import ResNet18

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['split_mnist', 'permuted_mnist'], default = 'split_cifar')
parser.add_argument('--n_tasks', type=int, default=5)
parser.add_argument('--n_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--buffer_batch_size', type=int, default=10)
parser.add_argument('--use_conv', action='store_true')
parser.add_argument('--samples_per_task', type=int, default=-1, help='if negative, full dataset is used')
parser.add_argument('--mem_size', type=int, default=600, help='controls buffer size') # mem_size in the tf repo.
parser.add_argument('--n_runs', type=int, default=1, help='number of runs to average performance')
parser.add_argument('--n_iters', type=int, default=1, help='training iterations on incoming batch')
parser.add_argument('--rehearsal', type=int, default=1, help='whether to replay previous data')
parser.add_argument('--hidden_dim', type=int, default=20)
parser.add_argument('--mixup', action='store_true', help='use manifold mixup')

# MbPA
parser.add_argument('--mbpa', action='store_true')
parser.add_argument('--mbpa_lr', type=float, default=0.1)
parser.add_argument('--mbpa_iters', type=int, default=5)
parser.add_argument('--mbpa_k', type=int, default=16)
args = parser.parse_args()

# fixed for now
args.input_size = (3, 32, 32)
args.device = 'cuda:0'
args.n_classes = 10
args.gen = False

# fetch data
data = locate('data.get_%s' % args.dataset)(args)

# make dataloaders
train_loader, test_loader  = [CLDataLoader(elem, args, train=t) for elem, t in zip(data, [True, False])]

# fetch model and ship to GPU
reset_model = lambda : ResNet18(args.n_classes, nf=args.hidden_dim).to(args.device)
reset_opt   = lambda model : torch.optim.SGD(model.parameters(), lr=0.1)

all_models = []


# Train the model 
# -------------------------------------------------------------------------------

for run in range(args.n_runs):
    model = reset_model()
    opt = reset_opt(model)

    # build buffer
    buffer = Buffer(args)
    buffer.min_per_class = 0 
    
    if run == 0: 
        print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
        print("buffer parameters:         ", np.prod(buffer.bx.size()))

    for task, loader in enumerate(train_loader):
        sample_amt = 0
        if task + 1 > args.n_tasks: break

        # iterate over samples from task
        for epoch in range(args.n_epochs):
            loss_ , correct, deno = 0., 0., 0.
            for i, (data, target) in enumerate(loader):
                if sample_amt > args.samples_per_task > 0: break
                sample_amt += data.size(0)
                
                data, target = data.to(args.device), target.to(args.device)
                
                for _ in range(args.n_iters):
                    train_idx, track_idx = buffer.split(args.buffer_batch_size if args.mixup else 0)
                    input_x, input_y = data, target

                    lamb = 1
                    hid = model.return_hidden(input_x)

                    if train_idx.nelement() > 0 and args.mixup:
                        lamb = np.random.beta(2, 2)
                        hid_b = model.return_hidden(buffer.bx[train_idx])
                        hid = lamb * hid + (1 - lamb) * hid_b
                     
                    logits = model.linear(hid)
                    loss_a = F.cross_entropy(logits, input_y, reduction='none')
                    loss   = loss_a.sum() 

                    if train_idx.nelement() > 0 and args.mixup: 
                        loss_b = F.cross_entropy(logits, buffer.by[train_idx], reduction='none')
                    else:
                        loss_b = 0

                    loss = (lamb * loss_a + (1 - lamb) * loss_b).sum() / loss_a.size(0)
                        
                    pred = logits.argmax(dim=1, keepdim=True)
                    correct += pred.eq(input_y.view_as(pred)).sum().item() 
                    deno  += pred.size(0)
                    loss_ += loss.item()
            
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    if track_idx.nelement() > 0 and args.rehearsal:
                        idx = track_idx[:args.buffer_batch_size]

                        mem_x, mem_y, logits_y = buffer.bx[idx], buffer.by[idx], None

                        opt.zero_grad()

                        logits_buffer = model(mem_x)
                        F.cross_entropy(logits_buffer, mem_y).backward()
                        opt.step()
                    
                # add data to reservoir
                buffer.add_reservoir(data, target, task)
            
            # buffer.display()
            print('Task {}\t Epoch {}\t Loss {:.6f}\t, Acc {:.6f}'.format(task, epoch, loss_ / i, correct / deno))

    all_models += [deepcopy(model)]


# Test the model 
# -------------------------------------------------------------------------------
avgs = []
with (torch.enable_grad if args.mbpa else torch.no_grad)():
    for model in all_models:
        model = model.eval()
        accuracies = []

        if args.mbpa:
            with torch.no_grad():
                key_gen = MBPA.init_keys(args)
                buffer.keys = key_gen(buffer.bx)

        for task, loader in enumerate(test_loader):
            # iterate over samples from task
            loss_, correct, deno = 0., 0., 0.
            for i, (data, target) in enumerate(loader):
                if args.mbpa and i > 2: break

                data, target = data.to(args.device), target.to(args.device)
                
                if args.mbpa:
                    with torch.no_grad():
                        data_keys = key_gen(data)
                        fetched_xy = MBPA.find_closest(buffer, data_keys, args)

                    pred = MBPA.predict(model, fetched_xy, data, args).unsqueeze(1)
                else:
                    logits = model(data)
                    pred = logits.argmax(dim=1, keepdim=True)
                
                correct += pred.eq(target.view_as(pred)).sum().item() 
                deno += data.size(0)
            
            accuracies += [correct / deno]

        out = ''
        for i, acc in enumerate(accuracies):
            out += '{} : {:.2f}\t'.format(i, acc)

        print(out)
        avgs += [sum(accuracies) / len(accuracies)]
        print('Avg','{:.5f}'.format(avgs[-1]), '\n')

print('\n\n\n AVG over {} runs : {:.4f}'.format(args.n_runs, sum(avgs) / len(avgs)))
