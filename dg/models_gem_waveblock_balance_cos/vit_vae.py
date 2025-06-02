from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import random
from collections import OrderedDict

from .metric import build_metric

#from timm.models import create_model
#import dg.models_gem_waveblock_balance_cos.vit_models

__all__ = ['VisionTransformer', 'vit_vae']
'''
class TransNetwork(nn.Module):
    def __init__(self):
        super(TransNetwork, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)

    def forward(self, x):
        out_1 = self.fc1(x) + x
        return out_1


'''
'''
class TransNetwork(nn.Module):
    def __init__(self):
        super(TransNetwork, self).__init__()
        self.b = nn.Parameter(torch.randn(4096))

    def forward(self, x):
        out_1 = x + self.b
        return out_1


'''
#'''
#best
class TransNetwork(nn.Module):
    def __init__(self, num_features):
        super(TransNetwork, self).__init__()
        self.fc1 = nn.Linear(4096, num_features, bias=False)

    def forward(self, x):
        out_1 = self.fc1(x)
        return out_1
#
#'''
'''
class TransNetwork(nn.Module):
    def __init__(self):
        super(TransNetwork, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU()

    def forward(self, x):
        #identity = x
        out_1 = self.relu(self.fc1(x)) + x
        #out_1 += identity 
        
        return out_1
'''     


'''
class TransNetwork(nn.Module):
    def __init__(self):
        super(TransNetwork, self).__init__()
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc5 = nn.Linear(2048, 4096)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out_1 = self.relu(self.fc1(x))
        out_2 = self.relu(self.fc2(out_1))
        out_3 = self.relu(self.fc3(out_2)+out_2)
        out_4 = self.relu(self.fc4(out_3)+out_1)
        
        out_5 = self.fc5(out_4)
        out_5 += identity 
        
        return out_5
'''


'''7layers
class TransNetwork(nn.Module):
    def __init__(self, num_features):
        super(TransNetwork, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 8192)
        self.fc4 = nn.Linear(8192, 8192)
        self.fc5 = nn.Linear(8192, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, num_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out_1 = self.relu(self.fc1(x))
        out_2 = self.relu(self.fc2(out_1))
        out_3 = self.relu(self.fc3(out_2))
        out_4 = self.relu(self.fc4(out_3)+out_3)
        out_5 = self.relu(self.fc5(out_4)+out_2)
        out_6 = self.relu(self.fc6(out_5)+out_1)
        
        out_7 = self.fc7(out_6)
        return out_7
'''

'''6layers
class TransNetwork(nn.Module):
    def __init__(self, num_features):
        super(TransNetwork, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 8192)
        self.fc4 = nn.Linear(8192, 4096)
        self.fc5 = nn.Linear(4096, 4096)
        self.fc6 = nn.Linear(4096, num_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out_1 = self.relu(self.fc1(x))
        out_2 = self.relu(self.fc2(out_1))
        out_3 = self.relu(self.fc3(out_2))
        out_4 = self.relu(self.fc4(out_3)+out_2)
        out_5 = self.relu(self.fc5(out_4)+out_1)
        
        out_6 = self.fc6(out_5)
        
        return out_6
'''  

'''5layers
class TransNetwork(nn.Module):
    def __init__(self, num_features):
        super(TransNetwork, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 8192)
        self.fc3 = nn.Linear(8192, 8192)
        self.fc4 = nn.Linear(8192, 4096)
        self.fc5 = nn.Linear(4096, num_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out_1 = self.relu(self.fc1(x))
        out_2 = self.relu(self.fc2(out_1))
        out_3 = self.relu(self.fc3(out_2)+out_2)
        out_4 = self.relu(self.fc4(out_3)+out_1)
        
        out_5 = self.fc5(out_4)
        
        return out_5
'''
'''4layers
class TransNetwork(nn.Module):
    def __init__(self, num_features):
        super(TransNetwork, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 8192)
        self.fc3 = nn.Linear(8192, 4096)
        self.fc4 = nn.Linear(4096, num_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out_1 = self.relu(self.fc1(x))
        out_2 = self.relu(self.fc2(out_1))
        out_3 = self.relu(self.fc3(out_2)+out_1)
        
        out_4 = self.fc4(out_3)
        
        return out_4
'''

'''3layers
class TransNetwork(nn.Module):
    def __init__(self, num_features):
        super(TransNetwork, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out_1 = self.relu(self.fc1(x))
        out_2 = self.relu(self.fc2(out_1)+out_1)
        
        out_3 = self.fc3(out_2)
        
        return out_3
'''
'''2layers
class TransNetwork(nn.Module):
    def __init__(self, num_features):
        super(TransNetwork, self).__init__()
        self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Linear(4096, num_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out_1 = self.relu(self.fc1(x))
        
        out_2 = self.fc2(out_1)
        
        return out_2
'''

class VisionTransformer(nn.Module):

    def __init__(self, weight, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 dev = None):
        super(VisionTransformer, self).__init__()
        self.pretrained = True
        self.weight = weight
        self.cut_at_pooling = cut_at_pooling
        self.num_classes = num_classes
        
        vit = TransNetwork(num_features).cuda()
            
       
        self.base = nn.Sequential(
            vit
        ).cuda()
        
        #self.linear = nn.Linear(512, 512)
        
        self.classifier = build_metric('cos', num_features, self.num_classes, s=64, m=0.35).cuda()
        #self.classifier_1 = build_metric('cos', 512, self.num_classes, s=64, m=0.6).cuda()
        
        self.projector_feat_bn = nn.Sequential(
                nn.Identity()
            ).cuda()

        '''
        self.projector_feat_bn_1 = nn.Sequential(
                self.linear,
                nn.Identity()
            ).cuda()
        '''

    def forward(self, x, y=None):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        
        bn_x = self.projector_feat_bn(x)
        prob = self.classifier(bn_x, y)
        
       # bn_x_512 = self.projector_feat_bn_1(bn_x)
        #prob_1 = self.classifier_1(bn_x_512, y)
        
        
        return bn_x, prob#, prob_1

def vit_vae(**kwargs):
    return VisionTransformer('base', **kwargs)