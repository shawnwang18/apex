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

import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                    help='Matrix M')

parser.add_argument('--input-size', type=int, default=2048, metavar='N',
                    help='Matrix K')

parser.add_argument('--hidden-size', type=int, default=2048, metavar='N',
                    help='Matrix K')

parser.add_argument('--output-size', type=int, default=2048, metavar='N',
                    help='Matrix N')

parser.add_argument('--bias', action='store_true', default=False,
                    help='whether use debug apex')

parser.add_argument('--logfile', type=str, default=None, help='logging output')

parser.add_argument('--hidden-layers', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--bucket-size', type=int, default=10000000, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--layers-per-bucket', type=int, default=0, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--delay-allreduce', action='store_true', default=False,
                    help='whether use delay allreduce')

parser.add_argument('--datatype', type=int, default=10000000, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--iteration-number', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--warmup-number', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--debug-apex', action='store_true', default=False,
                    help='whether use debug apex')

parser.add_argument('--debug-apex-dir', type=str, default='/home/scratch.shawnw_gpu/docker/apex/apex/parallel', 
                    help='custom-apex-dir')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument("--local_rank", default=0, type=int)

args = parser.parse_args()

if args.debug_apex:
  import sys
  sys.path.insert(0, args.debug_apex_dir)
  from distributed import DistributedDataParallel as DDP
else:
  from apex.parallel import DistributedDataParallel as DDP
  
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

def rank0print(*nargs):
 print_str = ""
 for item in nargs:
   print_str += str(item)
 if args.distributed:
   if torch.distributed.get_rank() == 0:
     print(print_str)
 else:
   print(print_str)


if args.distributed:
    assert args.cuda, "Distributed mode requires running with CUDA."
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    if torch.distributed.get_rank() == 0:
      if args.logfile == None:
        logfile = "logfile_rank0.log"
        args.logfile = logfile
      logging.basicConfig(format='%(filename)s:%(lineno)d:%(levelname)s:%(message)s', filename=args.logfile, level=logging.DEBUG)
    else:
      logging.basicConfig(format='%(filename)s:%(lineno)d:%(levelname)s:%(message)s', level=logging.INFO)
else:
    world_size = 1

rank0print("world_size:", world_size)

class Net(nn.Module):
    def __init__(self, args):
      super(Net, self).__init__()

      self.layers = []
      self.args = args

      #input layer
      layer = torch.nn.Linear(args.input_size, args.hidden_size, args.bias)
      self.add_module("input_layer", layer)
      self.layers.append(layer)
      
      #hidden layer
      for i in range(args.hidden_layers):
        layer = torch.nn.Linear(args.hidden_size, args.hidden_size, args.bias)
        self.add_module("hidden_layer_"+str(i), layer)
        self.layers.append(layer)

      #output layer
      layer = torch.nn.Linear(args.hidden_size, args.output_size, args.bias)
      self.add_module("output_layer", layer)
      self.layers.append(layer)

    def forward(self, x):
      for layer in self.layers:
        x = layer(x)
      return x

    def register_hook(self):
      self.grad_accs = []
      for name, param in self.named_parameters():
        if param.requires_grad:
          def wrapper(param, name):
            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            rank0print(grad_acc)
            def allreduce_hook(*unused):
              rank0print("running param:",name,  " grad at time:", time.time())
            grad_acc.register_hook(allreduce_hook)
            self.grad_accs.append(grad_acc)
          wrapper(param, name)

model = Net(args)

if args.cuda:
  model = model.cuda()

if args.debug_apex:
  model.register_hook()

if args.distributed:
  if args.layers_per_bucket != 0:
    args.bucket_size = args.layers_per_bucket * args.hidden_size *args.hidden_size 
  model = DDP(model,message_size=args.bucket_size,delay_allreduce=args.delay_allreduce)

def train():
    
  rank0print("**************************")
  rank0print(model)
  rank0print("**************************")
  for name, param in model.named_parameters():
    rank0print(name, " : ", param.size())
  rank0print("**************************")

  model.train()

  x = torch.randn(args.batch_size, args.input_size, requires_grad=False) 
  target = torch.randn(args.batch_size, args.output_size, requires_grad=False) 
  loss_fn = torch.nn.MSELoss()

  total_backward_time = 0
  total_e2e_time = 0
  total_loss_time = 0
  total_forward_time = 0

  if args.cuda:
    x = x.cuda()
    target = target.cuda()
    loss_fn = loss_fn.cuda()

  forward_start = torch.cuda.Event(enable_timing=True)
  forward_end   = torch.cuda.Event(enable_timing=True)
  nccl_end = torch.cuda.Event(enable_timing=True)

  compute_stream = torch.cuda.current_stream()

  pipeline_start = True

  for i in range(args.iteration_number):

    rank0print("===ITERATION:%s====", i)

    if (pipeline_start != True):
        grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]
        buckets = split_by_type(grads) 
        for tp in buckets:
            bucket = buckets[tp]
        coalesced = flatten(bucket)
        if extra_args is not None:
            call(coalesced, *extra_args)
        else:
            call(coalesced)
        if call is dist.all_reduce:
            coalesced /= dist.get_world_size()
                                                      
        for buf, synced in zip(bucket, unflatten(coalesced, bucket)):
            buf.copy_(synced)

    model.zero_grad()      

    forward_start_time = time.time()
    output = model(x)
    compute_stream.synchronize()
    forward_end_time = time.time()

    loss = loss_fn(output, target)
    compute_stream.synchronize()
    loss_end_time = time.time()

    loss.backward()
    compute_stream.synchronize()

    backward_end_time = time.time()

    rank0print("step ", i, 
        ",e2e:", backward_end_time-forward_start_time, 
        ",forward:", forward_end_time-forward_start_time, 
        ",loss:",loss_end_time - forward_end_time,
        ",backward:",backward_end_time - loss_end_time)

    if i+1 > args.warmup_number:
      total_e2e_time += backward_end_time - forward_start_time
      total_backward_time += backward_end_time - loss_end_time
      total_loss_time += loss_end_time - forward_end_time
      total_forward_time += forward_end_time-forward_start_time
  
  average_e2e_time = total_e2e_time/(args.iteration_number-args.warmup_number)
  average_forward_time = total_forward_time/(args.iteration_number-args.warmup_number)
  average_loss_time = total_loss_time/(args.iteration_number-args.warmup_number)
  average_backward_time = total_backward_time/(args.iteration_number-args.warmup_number)

  rank0print("===RUN EPIOLOG===")
  rank0print("Average e2e:", average_e2e_time,
            ",forward:", average_forward_time,
            ",loss:", average_loss_time,
            ",backward:", average_backward_time)

  rank0print("csv;{};{};{};{};{};{}".format(world_size, args.layers_per_bucket, average_e2e_time, average_forward_time, average_loss_time, average_backward_time))

train()

logging.shutdown()
