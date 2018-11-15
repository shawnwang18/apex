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

parser.add_argument('--logdir', type=str, default="./", help='logging output')

parser.add_argument('--hidden-layers', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--bucket-size', type=int, default=10000000, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--layers-per-bucket', type=int, default=0, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--delay-allreduce', action='store_true', default=False,
                    help='whether use delay allreduce')

parser.add_argument('--logging', action='store_true', default=False,
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
    
    if args.logging: 
      logfile = args.logdir + "/logfile_rank_" + str(torch.distributed.get_rank()) +"_.log"
      args.logfile = logfile
      logging.basicConfig(format='%(filename)s:%(lineno)d:%(levelname)s:%(message)s', filename=args.logfile, filemode='w', level=logging.DEBUG)

    #if torch.distributed.get_rank() == 0:
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
                logging.debug("param grad; %s;%s",name, time.time())
            grad_acc.register_hook(allreduce_hook)
            self.grad_accs.append(grad_acc)
          wrapper(param, name)

model = Net(args)

if args.cuda:
  model = model.cuda()

#if args.debug_apex:
#  model.register_hook()

if args.distributed:
  if args.layers_per_bucket != 0:
    args.bucket_size = args.layers_per_bucket * args.hidden_size *args.hidden_size 
  model = DDP(model,message_size=args.bucket_size,delay_allreduce=args.delay_allreduce)
  #model = DDP(model,allreduce_trigger_params=None if args.delay_allreduce else model.parameters(),delay_allreduce=args.delay_allreduce)

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

  for i in range(args.iteration_number):

    rank0print("===ITERATION:%s====", i)
    model.zero_grad()      

    forward_start_time = time.time()
    output = model(x)
    torch.cuda.synchronize()
    forward_end_time = time.time()

    loss = loss_fn(output, target)
    compute_stream.synchronize()
    loss_end_time = time.time()

    loss.backward()
    torch.cuda.synchronize()
    backward_end_time = time.time()

    if args.distributed:
      model.timing_log()

    rank0print("step ", i, 
        ",e2e:", (backward_end_time-forward_start_time)*1000, 
        ",forward:", (forward_end_time-forward_start_time)*1000, 
        ",loss:",(loss_end_time - forward_end_time)*1000,
        ",backward:",(backward_end_time - loss_end_time)*1000)

    if i+1 > args.warmup_number:
      total_e2e_time += (backward_end_time - forward_start_time)*1000
      total_backward_time += (backward_end_time - loss_end_time)*1000
      total_loss_time += (loss_end_time - forward_end_time)*1000
      total_forward_time += (forward_end_time-forward_start_time)*1000
  
  average_e2e_time = total_e2e_time/(args.iteration_number-args.warmup_number)
  average_forward_time = total_forward_time/(args.iteration_number-args.warmup_number)
  average_loss_time = total_loss_time/(args.iteration_number-args.warmup_number)
  average_backward_time = total_backward_time/(args.iteration_number-args.warmup_number)

  rank0print("===RUN EPIOLOG===")
  rank0print("Average e2e:", average_e2e_time,
            ",forward:", average_forward_time,
            ",loss:", average_loss_time,
            ",backward:", average_backward_time)

  #rank0print("csv;{};{};{};{};{};{}".format(world_size, args.layers_per_bucket, average_e2e_time, average_forward_time, average_loss_time, average_backward_time))

  if not args.distributed:
    rank0print("task_csv;{};{};{};{};{};{}".format(
      world_size, 
      args.layers_per_bucket, 
      average_e2e_time,
      0,
      0,
      0))
  elif args.logging:
    time.sleep(60)
    if (args.local_rank == 0):
    
      import glob
      import csv
      
      logfile = args.logdir + "/logfile_rank_*.log"
      files=glob.glob(logfile)
      
      total_rank = len(files)
      bucket_reduce_time = {}
      
      table = {}
      for file in files:
          filename = os.path.basename(file)
          file_name_split = filename.split('_')
          rank_id = int(file_name_split[2])
          source_file = open(file, "r")
      
          iteration_idx = 1
          bucket_reduce_time[rank_id] = []
      
          for line in source_file:
            
            new_line = line.rstrip()
            line_split = new_line.split(';')

            if (iteration_idx + 1 > args.warmup_number):
              if 'bucket_reduce' in line:
                bucket_reduce_time[rank_id].append(['reduce',iteration_idx, line_split[1], float(line_split[2])])
              elif 'fwd-compute-epilog' in line:
                bucket_reduce_time[rank_id].append(['forward', line_split[1], '', float(line_split[2])])
                bucket_reduce_time[rank_id].append(['maintime',line_split[1], '', float(line_split[3])])
                bucket_reduce_time[rank_id].append(['epilog',  line_split[1], '', float(line_split[4])])
                bucket_reduce_time[rank_id].append(['EffLoss',  line_split[1], '',   float(line_split[3])/(float(line_split[3])+float(line_split[4]))])

            if 'fwd-compute-epilog' in line:
              iteration_idx += 1
      
      for rank_id in range(total_rank):
        if rank_id > 0:
          for i in range(len(bucket_reduce_time[rank_id])):
            bucket_reduce_time[0][i].append(bucket_reduce_time[rank_id][i][3])
     
      skew_list = []
      epilog_list = []
      eff_list = []
      compute_list = []
      for item in bucket_reduce_time[0]:
        if 'epilog' in item:
          epilog_list.append(item)
          min_time = item[3]
          max_time = item[3]
          for temp_time in item[3:]:
            if temp_time < min_time:
              min_time = temp_time
            if temp_time > max_time:
              max_time = temp_time
          skew = max_time - min_time
          skew_list.append(['skew', item[1], min_time, max_time, skew])
        elif 'EffLoss' in item:
          eff_list.append(item)
        elif 'maintime' in item:
          compute_list.append(item)
    
      avg_skew = sum([item[4] for item in skew_list])/len(skew_list)
      avg_epilog_list = [sum(item[3:])/len(item[3:]) for item in epilog_list]
      avg_epilog = sum(avg_epilog_list)/len(avg_epilog_list)
      avg_eff_list = [sum(item[3:])/len(item[3:]) for item in eff_list]
      avg_eff = sum(avg_eff_list)/len(avg_eff_list)
      avg_compute_list = [sum(item[3:])/len(item[3:]) for item in compute_list]
      avg_compute = sum(avg_compute_list)/len(avg_compute_list)
    
      rank0print("task_csv;{};{};{};{};{};{}".format(
        world_size, 
        args.layers_per_bucket, 
        avg_compute,
        avg_epilog,
        avg_skew,
        avg_eff))
      
      write_file = open(os.path.join(args.logdir, "parsed.csv"),"w")
      for item in bucket_reduce_time[0]:
          write_file.write(";".join([str(sli) for sli in item ])+"\n")
      write_file.close()
    
      write_file = open(os.path.join(args.logdir, "skew.csv"),"w")
      for item in skew_list:
          write_file.write(";".join([str(sli) for sli in item ])+"\n")
      write_file.close()

      write_file = open(os.path.join(args.logdir, "epilog.csv"),"w")
      for item in epilog_list:
          write_file.write(";".join([str(sli) for sli in item ])+"\n")
      write_file.close()

      write_file = open(os.path.join(args.logdir, "maintime.csv"),"w")
      for item in compute_list:
          write_file.write(";".join([str(sli) for sli in item ])+"\n")
      write_file.close()
      

train()

logging.shutdown()
