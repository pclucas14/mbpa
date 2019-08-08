import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from copy  import deepcopy
from model import ResNet18



# create all the models
class megamodel(nn.Module):
    def __init__(self):
        super(megamodel, self).__init__()
        self.all_models = nn.ModuleList([ResNet18(50, 20) for _ in range(5)])

    def forward(self, x):
        out = torch.cat([model(x) for model in self.all_models], dim=1)
        return out


def fetch_key_network(model, args):
    if args.key_network == 'random_ensemble':
        key_gen = megamodel().to(args.device)
        return key_gen
    elif args.key_network == 'pretrained':
        from torchvision import models 
        resnet = models.resnet18(pretrained=True).to('cuda')
        return lambda x : resnet(F.upsample(x, scale_factor=2.))
    elif args.key_network == 'hidden':
        return model.return_hidden


def find_closest(buffer, data_keys, args):
    buffer_keys = buffer.keys.unsqueeze(0)
    data_keys = data_keys.unsqueeze(1)
    dist = torch.norm( (buffer_keys - data_keys), 2, 2)
    _, min_idx = dist.topk(k=args.mbpa_k, largest=False)

    most_interfered = buffer.bx[min_idx], buffer.by[min_idx]
    return most_interfered


def predict(model, fetched_xy, data, args):
    all_preds = []
    
    for i in range(fetched_xy[0].size(0)):
        mir_x, mir_y = fetched_xy[0][i], fetched_xy[1][i]
        
        finetune  = deepcopy(model)
        opt = torch.optim.SGD(finetune.parameters(), lr=args.mbpa_lr)

        for step in range(args.mbpa_iters):
            opt.zero_grad()
            F.cross_entropy(finetune(mir_x), mir_y).backward()
            opt.step()

        pred = finetune(data[[i]])
        all_preds += [pred.argmax(dim=1)]

    return torch.cat(all_preds) 

    


