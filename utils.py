import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.nn.utils.weight_norm as wn
from collections import OrderedDict as OD
from collections import defaultdict as DD

# good'ol utils
# ---------------------------------------------------------------------------------
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def print_and_save_args(args, path):
    print(args)
    # let's save the args as json to enable easy loading
    with open(os.path.join(path, 'args.json'), 'w') as f: 
        json.dump(vars(args), f)

def load_model_from_file(path):
    with open(os.path.join(path, 'args.json'), 'rb') as f:
        args = dotdict(json.load(f))

    from buffer import Buffer
    from model  import ResNet18
    
    # create model
    model = ResNet18(args.n_classes, nf=args.hidden_dim).to(args.device)
    buffer = Buffer(args)

    # load weights
    model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
    buffer.load_state_dict(torch.load(os.path.join(path, 'buffer.pth')))

    return model, buffer

def save_model_to_file(model, buffer, args): 
    # let's save the args as json to enable easy loading
    with open(os.path.join(args.path, 'args.json'), 'w') as f: 
        json.dump(vars(args), f)

    torch.save(model.state_dict(), os.path.join(args.path, 'model.pth'))
    torch.save(buffer.state_dict(), os.path.join(args.path, 'buffer.pth'))
    

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_and_log_scalar(writer, name, value, write_no, end_token=''):
    if isinstance(value, list):
        if len(value) == 0: return
        
        str_tp = str(type(value[0]))
        if type(value[0]) == torch.Tensor:
            value = torch.mean(torch.stack(value))
        elif 'float' in str_tp or 'int' in str_tp:
            value = sum(value) / len(value)
    zeros = 40 - len(name) 
    name += ' ' * zeros
    print('{} @ write {} = {:.4f}{}'.format(name, write_no, value, end_token))
    writer.add_scalar(name, value, write_no)

def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# create logging containers
def reset_log():
    return DD(list)

