import math
import logging
import numpy as np

import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F

from nearest_embed import NearestEmbed

# Generative Models 
# -----------------------------------------------------------------------------------

""" VQ-VAE (credit: https://github.com/nadavbh12/VQ-VAE) """
class VQ_CVAE(nn.Module):
    def __init__(self, d, k=10, bn=True, num_channels=3, **kwargs):
        super(VQ_CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),

            # makes latents 4 x 4 instead of 8 x 8 
            # nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(d),
            # nn.ReLU(inplace=True),
            
            ResBlock(d, d, bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn),
            nn.BatchNorm2d(d),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            
            #  makes latents 4 x 4 instead of 8 x 8 
            # nn.BatchNorm2d(d),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),#0),

            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, num_channels, kernel_size=4, stride=2, padding=1),
        )
        
        self.d = d
        self.emb = NearestEmbed(k, d)

        # add classifier
        self.classifier = nn.Sequential(
                # ResBlock(d, d), 
                # nn.BatchNorm2d(d), 
                # ResBlock(d, d),
                nn.Conv2d(d, 2*d, 4, 2, 1), 
                nn.BatchNorm2d(2*d), 
                nn.Conv2d(2*d, 4*d, 4, 2, 1))
        self.classifier_out = nn.Linear(1600, 10)

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def classify(self, hid):
        hid = self.classifier(hid)
        return self.classifier_out(hid.view(hid.size(0), -1))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return F.tanh(self.decoder(x))

    def forward(self, x, masked_indices=None):
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, self.training, masked_indices, weight_sg=True)
        emb, _ = self.emb(z_e.detach(), self.training, masked_indices)
        return self.decode(z_q), self.classify(z_q), z_e, emb, argmin

    def sample(self, size):
        with torch.no_grad():
            sample = torch.randn(size, self.d, self.f, self.f)
            if self.cuda():
                sample = sample.cuda()
            emb, _ = self.emb(sample)
            return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        mse = F.mse_loss(recon_x, x)
        vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
        commit_loss = torch.mean(torch.norm((emb.detach() - z_e)**2, 2, 1))

        return mse, vq_loss, commit_loss

    def print_atom_hist(self, argmin):

        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)

class ResBlock(nn.Module):
    def __init__(self, in_channels, channels, bn=False):
        super(ResBlock, self).__init__()

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)



# Classifiers 
# -----------------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def return_hidden(self, x):
        bsz = x.size(0)
        out = F.relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.return_hidden(x)
        out = self.linear(out)
        return out

def ResNet18(nclasses, nf=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)
