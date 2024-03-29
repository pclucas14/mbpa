import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import mbpa as MBPA

from data   import *
from utils  import * 
from buffer import Buffer
from copy   import deepcopy
from pydoc  import locate
from model  import ResNet18

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
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

parser.add_argument('--load_path', type=str)
parser.add_argument('--save_model', action='store_true')

# MbPA
parser.add_argument('--mbpa', action='store_true')
parser.add_argument('--mbpa_lr', type=float, default=0.1)
parser.add_argument('--mbpa_iters', type=int, default=5)
parser.add_argument('--mbpa_k', type=int, default=16)
parser.add_argument('--key_network', type=str, choices=['pretrained', 'random_ensemble', 'hidden'], default='random_ensemble')

# Meta Learning Representaions
parser.add_argument('--meta', action='store_true')
parser.add_argument('--meta_iters', type=int, default=3)
parser.add_argument('--meta_lr', type=float, default=0.2)

args = parser.parse_args()

# fixed for now
args.input_size = (3, 32, 32)
args.device = 'cuda:0'
args.n_classes = 10
args.gen = False
args.path = 'runs/M:{}_DS:{}_NI:{}{}'.format(args.mem_size, args.dataset[:10], args.n_iters, '-meta' if args.meta else '')
maybe_create_dir(args.path)
print(args.path)

# fetch data
data = locate('data.get_%s' % args.dataset)(args)

# make dataloaders
train_loader, test_loader  = [CLDataLoader(elem, args, train=t) for elem, t in zip(data, [True, False])]

# fetch model and ship to GPU
reset_model = lambda : ResNet18(args.n_classes, nf=args.hidden_dim).to(args.device)
reset_opt   = lambda model : torch.optim.SGD(model.parameters(), lr=args.lr)

all_models = []

def average_weights(model_before, model_after):
    # we do the update on `model_before` and discard `model_after`
    weights_before, weights_after = model_before.state_dict(), model_after.state_dict()

    model_before.load_state_dict({name : 
                    weights_before[name] + (weights_after[name] - weights_before[name]) * args.meta_lr
                    for name in weights_before})

    return model_before


if args.load_path is None:
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
                        train_idx, track_idx = buffer.split(0)
                        input_x, input_y = data, target

                        lamb = 1
                        hid = model.return_hidden(input_x)

                        logits = model.linear(hid)
                        loss   = F.cross_entropy(logits, input_y)
                            
                        pred = logits.argmax(dim=1, keepdim=True)
                        correct += pred.eq(input_y.view_as(pred)).sum().item() 
                        deno  += pred.size(0)
                        loss_ += loss.item()
                
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        if task > 0 and args.rehearsal:

                            """ meta training """ 
                            if args.meta:
                                # task specific sampling 
                                #mem_x, mem_y = buffer.sample_from_task(args.buffer_batch_size, 
                                #                                       np.random.randint(task))
                                # regular sampling 
                                idx = track_idx[:args.buffer_batch_size]
                                mem_x, mem_y = buffer.bx[idx], buffer.by[idx]

                                tmp_model = deepcopy(model)
                                tmp_opt   = reset_opt(tmp_model)
                                for _ in range(args.meta_iters):
                                    tmp_opt.zero_grad()
                                    logits_buffer = tmp_model(mem_x)
                                    F.cross_entropy(logits_buffer, mem_y).backward()
                                    tmp_opt.step()

                                # average the models
                                model = average_weights(model, tmp_model)

                            else:
                                # regular sampling 
                                idx = track_idx[:args.buffer_batch_size]
                                mem_x, mem_y = buffer.bx[idx], buffer.by[idx]
                                opt.zero_grad()
                                logits_buffer = model(mem_x)
                                F.cross_entropy(logits_buffer, mem_y).backward()
                                opt.step()
                        
                    # add data to reservoir
                    buffer.add_reservoir(data, target, task)
                
                # buffer.display()
                print('Task {}\t Epoch {}\t Loss {:.6f}\t, Acc {:.6f}'.format(task, epoch, loss_ / i, correct / deno))

        all_models += [deepcopy(model)]

        if run == 0 and args.save_model:
            save_model_to_file(model, buffer, args)
    
else:
    model, buffer = load_model_from_file(args.load_path)
    all_models += [model]


# Test the model 
# -------------------------------------------------------------------------------
avgs = []
with (torch.enable_grad if args.mbpa else torch.no_grad)():
    for model in all_models:
        model = model.eval()
        accuracies = []

        if args.mbpa:
            with torch.no_grad():
                # leverage untrained models ?
                key_gen = MBPA.fetch_key_network(model, args)

                # leverage the trained model ?
                # key_gen  = model.return_hidden
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
