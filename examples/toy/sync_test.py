from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from apex.fp16_utils import to_python_float
import pprint
import logging
import time

#=====START: ADDED FOR DISTRIBUTED======
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
#=====END:   ADDED FOR DISTRIBUTED======

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--M', type=int, default=8192, metavar='N',
                    help='Matrix M')

parser.add_argument('--N', type=int, default=8192, metavar='N',
                    help='Matrix K')

parser.add_argument('--K', type=int, default=8192, metavar='N',
                    help='Matrix K')

parser.add_argument('--allreduce-size', type=int, default=536870912, metavar='N',
                    help='Matrix N')

parser.add_argument('--datatype', type=str, default="float", help='String')

parser.add_argument('--bias', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument("--local_rank", default=0, type=int)

args = parser.parse_args()

torch.cuda.set_device(args.local_rank)

#torch.distributed.init_process_group(backend='nccl', init_method='env://')

#from apex.parallel.distributed import flat_dist_call

#def rank0print(*nargs):
#  if torch.distributed.get_rank() == 0:
#    print_str = ""
#    for item in nargs:
#      print_str += str(item)
#    print(print_str)
#
#world_size = torch.distributed.get_world_size()
#rank0print("global_world_size:", world_size)

def nccl_gem_interference(M = 8192, N = 8192, K=8192, allreduce_size = 536870912, datatype = "float", bias=False):

  x = torch.randn(M, K, requires_grad=False)
  y = torch.randn(M, K, requires_grad=False)

  layer1 = torch.nn.Linear(K, N, bias).cuda()
  layer2 = torch.nn.Linear(K, N, bias).cuda()

  x = x.cuda()
  y = y.cuda()

  z = layer1(x)
  torch.cuda.current_stream().synchronize()
  layer2(y)
  torch.cuda.current_stream().synchronize()

nccl_gem_interference(M=4096  , N=4096  , K=4096  , allreduce_size = 16777216  , datatype = "half" , bias = False)
