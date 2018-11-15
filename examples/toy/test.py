from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import partial
from torchvision import datasets, transforms
from torch.autograd import Variable
from apex.fp16_utils import to_python_float
import pprint
import logging
import time
import csv


import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
#=====END:   ADDED FOR DISTRIBUTED======

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--csv-dir', type=str, default='./logs', 
                    help='csv file for logging output')

parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--run-iterations', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--warmup-iterations', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 64)')

args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
from apex.parallel.distributed import flat_dist_call



def nccl_gem_interference(csv_writer = None, M = 8192, N = 8192, K=8192, allreduce_size = 536870912, datatype = "float", bias=False):

  x           = torch.randn(M, K, requires_grad = False)
  weights     = torch.randn(allreduce_size, requires_grad = False)
 
  gemm_stream = torch.cuda.current_stream()
  nccl_stream = torch.cuda.Stream()

  gemm_op     = torch.nn.Linear(K, N, bias).cuda()

  x = x.cuda()	
  weights = weights.cuda()
   
  with torch.cuda.stream(gemm_stream):
  	flat_dist_call(weights,dist.all_reduce)
  
  torch.cuda.synchronize()
  with torch.cuda.stream(nccl_stream):
  	flat_dist_call(weights,dist.all_reduce)

  torch.cuda.synchronize()


nccl_gem_interference()
