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

#=====START: ADDED FOR DISTRIBUTED======
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


def rank0print(*nargs):
  if torch.distributed.get_rank() == 0:
    print_str = ""
    for item in nargs:
      print_str += str(item)
    print(print_str)

world_size = torch.distributed.get_world_size()
rank0print("global_world_size:", world_size)

def interference_test(op1, stream1, op2, stream2,csv_cfg):

  csv_writer = csv_cfg['file']

  start = torch.cuda.Event(enable_timing=True)
  op1_end = torch.cuda.Event(enable_timing=True)
  op2_end = torch.cuda.Event(enable_timing=True)
  
  torch.cuda.synchronize()

  #Runing 2 streams 
  total_itr_2stream_time = 0
  total_op1_2stream_time = 0
  total_op2_2stream_time = 0
  for i in range(args.run_iterations):
    stream1.synchronize()
    stream2.synchronize()
    stream1.record_event(start)
    with torch.cuda.stream(stream1):
      op1()
      op1_end.record(stream1)
    with torch.cuda.stream(stream2):
      op2()
      op2_end.record(stream2)
    #stream1.record_event(op1_end)
    #stream2.record_event(op2_end)
    op1_end.synchronize()
    op2_end.synchronize()
    #torch.cuda.synchronize()
    op1_2stream_time = start.elapsed_time(op1_end)
    op2_2stream_time = start.elapsed_time(op2_end)
    itr_2stream_time = op1_2stream_time if op1_2stream_time > op2_2stream_time else op2_2stream_time

    if i+1 > args.warmup_iterations:
      total_itr_2stream_time += itr_2stream_time
      total_op1_2stream_time += op1_2stream_time
      total_op2_2stream_time += op2_2stream_time

      rank0print("iteration",i,"_run_2stream_time: ", itr_2stream_time)
      rank0print("iteration",i,"_op1_2stream_time: ", op1_2stream_time)
      rank0print("iteration",i,"_op2_2stream_time: ", op2_2stream_time)

    average_itr_2stream_time = total_itr_2stream_time/(args.run_iterations - args.warmup_iterations)
    average_op1_2stream_time = total_op1_2stream_time/(args.run_iterations - args.warmup_iterations)
    average_op2_2stream_time = total_op2_2stream_time/(args.run_iterations - args.warmup_iterations)

  #Runing 1 streams 
  total_itr_1stream_time = 0
  total_op1_1stream_time = 0
  total_op2_1stream_time = 0
  start = torch.cuda.Event(enable_timing=True)
  op1_end = torch.cuda.Event(enable_timing=True)
  op2_end = torch.cuda.Event(enable_timing=True)

  for i in range(args.run_iterations):

    torch.cuda.synchronize()
    stream1.synchronize()
    with torch.cuda.stream(stream1):
      stream1.record_event(start)
      op1()
      stream1.record_event(op1_end)
      op1_end.synchronize()
      op1_1stream_time = start.elapsed_time(op1_end)

    torch.cuda.synchronize()
    with torch.cuda.stream(stream1):
      stream1.record_event(start)
      op2()
      stream1.record_event(op2_end)
      op2_end.synchronize()
      op2_1stream_time = start.elapsed_time(op2_end)

    itr_1stream_time = op1_1stream_time + op2_1stream_time
    torch.cuda.synchronize()

    if i+1 > args.warmup_iterations:
      total_itr_1stream_time += itr_1stream_time
      total_op1_1stream_time += op1_1stream_time
      total_op2_1stream_time += op2_1stream_time

      rank0print("iteration",i,"_run_1stream_time: ", itr_1stream_time)
      rank0print("iteration",i,"_op1_1stream_time: ", op1_1stream_time)
      rank0print("iteration",i,"_op2_1stream_time: ", op2_1stream_time)

    average_itr_1stream_time = total_itr_1stream_time/(args.run_iterations - args.warmup_iterations)
    average_op1_1stream_time = total_op1_1stream_time/(args.run_iterations - args.warmup_iterations)
    average_op2_1stream_time = total_op2_1stream_time/(args.run_iterations - args.warmup_iterations)

  csv_line = csv_cfg['op_cfg'] + [average_itr_2stream_time, average_op1_2stream_time, average_op2_2stream_time, average_itr_1stream_time, average_op1_1stream_time, average_op2_1stream_time]
  csv_writer.writerow(csv_line)

def op_test(op, stream):
  op_start = torch.cuda.Event(enable_timing=True)
  op_end = torch.cuda.Event(enable_timing=True)
  torch.cuda.synchronize()

  total_op_time = 0
  for i in range(args.run_iterations):
    with torch.cuda.stream(stream):
      stream.record_event(op_start)
      op()
      stream.record_event(op_end)
    op_end.synchronize()
    torch.cuda.synchronize()

    itr_op_time = op_start.elapsed_time(op_end)

    if i+1 > args.warmup_iterations:
      total_op_time += itr_op_time
      rank0print("iteration",i,"time: ", itr_op_time)

    average_itr_time = total_op_time/(args.run_iterations - args.warmup_iterations)

    return [average_itr_time]

def nccl_gem_interference(csv_writer = None, M = 8192, N = 8192, K=8192, allreduce_size = 536870912, datatype = "float", bias=False):
  x           = torch.randn(M, K, requires_grad = False)
  weights     = torch.randn(allreduce_size, requires_grad = False)
  gemm_stream = torch.cuda.current_stream()
  nccl_stream = torch.cuda.Stream()
  gemm_op     = torch.nn.Linear(K, N, bias).cuda()

  if datatype == "half":
    x = x.half()
    weights = weights.half()
    gemm_op = gemm_op.half()

  x = x.cuda()
  weights = weights.cuda()

  output = gemm_op(x)
  loss = output.mean()

  op1 = partial(loss.backward, retain_graph=True)
  op2 = partial(flat_dist_call, [weights],dist.all_reduce)

  csv_cfg           = {}
  csv_cfg['op_cfg'] = [M,N,K,allreduce_size, datatype, bias]
  csv_cfg['file']   = csv_writer

  rank0print("===GEMM_NCCL_TEST:", csv_cfg['op_cfg'])
  interference_test(op1, gemm_stream, op2, nccl_stream, csv_cfg)

def nccl_conv2d_interference(csv_writer = None, allreduce_size = 536870912, 
    n_in = 256, c_in = 1024, h_in = 1024, w_in = 1024, 
    c_out = 1024, kernel=16, datatype = "float", bias=False):

  x = torch.randn(n_in, c_in, h_in, w_in, requires_grad = False)
  weights     = torch.randn(allreduce_size, requires_grad = False)

  conv_stream = torch.cuda.current_stream()
  nccl_stream = torch.cuda.Stream()

  conv_op = torch.nn.Conv2d(c_in, c_out, kernel, bias=bias).cuda()

  if datatype == "half":
    x = x.half()
    weights = weights.half()
    conv_op = conv_op.half()

  x = x.cuda()
  weights = weights.cuda()

  output = conv_op(x)
  loss = output.mean()

  op1 = partial(loss.backward, retain_graph=True)
  op2 = partial(flat_dist_call, [weights], dist.all_reduce)

  csv_cfg           = {}
  csv_cfg['op_cfg'] = [allreduce_size, n_in, c_in, h_in, w_in, c_out, kernel]
  csv_cfg['file']   = csv_writer

  rank0print("===CONV2D_NCCL_TEST:", csv_cfg['op_cfg'])
  interference_test(op1, conv_stream, op2, nccl_stream, csv_cfg)

def gem_nccl_tests():

  csv_file = os.path.join(args.csv_dir, "gem_allreduce.csv") 
  csv_file = open(csv_file, 'w') 
  csv_writer = csv.writer(csv_file)
  csv_header = ["M","N","K","ReduceSize", "Type", "Bias"] + ["overlap_time", "overlap_op1_time", "overlap_op2_time", "seri_time", "seri_op1_time", "seri_op2_time"]
  csv_writer.writerow(csv_header)

  ##nccl_gem_interference(csv_writer , M=256   , N=256   , K=256   , allreduce_size = 16777216    , datatype = "half"  , bias = False)
  ##nccl_gem_interference(csv_writer , M=512   , N=512   , K=512   , allreduce_size = 16777216    , datatype = "half"  , bias = False)
  ##nccl_gem_interference(csv_writer , M=1024  , N=1024  , K=1024  , allreduce_size = 16777216    , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer , M=2048  , N=2048  , K=2048  , allreduce_size = 16777216    , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=4096  , N=4096  , K=4096  , allreduce_size = 16777216    , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=8192  , N=8192  , K=8192  , allreduce_size = 16777216    , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=16384 , N=16384 , K=16384 , allreduce_size = 16777216    , datatype = "half"  , bias = False)

  ##nccl_gem_interference(csv_writer , M=256   , N=256   , K=256   , allreduce_size = 33554432    , datatype = "half"  , bias = False)
  ##nccl_gem_interference(csv_writer , M=512   , N=512   , K=512   , allreduce_size = 33554432    , datatype = "half"  , bias = False)
  ##nccl_gem_interference(csv_writer , M=1024  , N=1024  , K=1024  , allreduce_size = 33554432    , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer , M=2048  , N=2048  , K=2048  , allreduce_size = 33554432    , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=4096  , N=4096  , K=4096  , allreduce_size = 33554432    , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=8192  , N=8192  , K=8192  , allreduce_size = 33554432    , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=16384 , N=16384 , K=16384 , allreduce_size = 33554432    , datatype = "half"  , bias = False)

  ##nccl_gem_interference(csv_writer , M=256   , N=256   , K=256   , allreduce_size = 67108864    , datatype = "half"  , bias = False)
  ##nccl_gem_interference(csv_writer , M=512   , N=512   , K=512   , allreduce_size = 67108864    , datatype = "half"  , bias = False)
  ##nccl_gem_interference(csv_writer , M=1024  , N=1024  , K=1024  , allreduce_size = 67108864    , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer , M=2048  , N=2048  , K=2048  , allreduce_size = 67108864    , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=4096  , N=4096  , K=4096  , allreduce_size = 67108864    , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=8192  , N=8192  , K=8192  , allreduce_size = 67108864    , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=16384 , N=16384 , K=16384 , allreduce_size = 67108864    , datatype = "half"  , bias = False)

  ##nccl_gem_interference(csv_writer , M=256   , N=256   , K=256   , allreduce_size = 134217728   , datatype = "half"  , bias = False)
  ##nccl_gem_interference(csv_writer , M=512   , N=512   , K=512   , allreduce_size = 134217728   , datatype = "half"  , bias = False)
  ##nccl_gem_interference(csv_writer , M=1024  , N=1024  , K=1024  , allreduce_size = 134217728   , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer , M=2048  , N=2048  , K=2048  , allreduce_size = 134217728   , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=4096  , N=4096  , K=4096  , allreduce_size = 134217728   , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=8192  , N=8192  , K=8192  , allreduce_size = 134217728   , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=16384 , N=16384 , K=16384 , allreduce_size = 134217728   , datatype = "half"  , bias = False)

  ##nccl_gem_interference(csv_writer , M=256   , N=256   , K=256   , allreduce_size = 134217728*2 , datatype = "half"  , bias = False)
  ##nccl_gem_interference(csv_writer , M=512   , N=512   , K=512   , allreduce_size = 134217728*2 , datatype = "half"  , bias = False)
  ##nccl_gem_interference(csv_writer , M=1024  , N=1024  , K=1024  , allreduce_size = 134217728*2 , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer , M=2048  , N=2048  , K=2048  , allreduce_size = 134217728*2 , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=4096  , N=4096  , K=4096  , allreduce_size = 134217728*2 , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=8192  , N=8192  , K=8192  , allreduce_size = 134217728*2 , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=16384 , N=16384 , K=16384 , allreduce_size = 134217728*2 , datatype = "half"  , bias = False)

  ##nccl_gem_interference(csv_writer , M=256   , N=256   , K=256   , allreduce_size = 134217728*4 , datatype = "half"  , bias = False)
  ##nccl_gem_interference(csv_writer , M=512   , N=512   , K=512   , allreduce_size = 134217728*4 , datatype = "half"  , bias = False)
  ##nccl_gem_interference(csv_writer , M=1024  , N=1024  , K=1024  , allreduce_size = 134217728*4 , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer , M=2048  , N=2048  , K=2048  , allreduce_size = 134217728*4 , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=4096  , N=4096  , K=4096  , allreduce_size = 134217728*4 , datatype = "half"  , bias = False)
  #nccl_gem_interference(csv_writer  , M=8192  , N=8192  , K=8192  , allreduce_size = 134217728*4 , datatype = "half"  , bias = False)
  nccl_gem_interference(csv_writer  , M=16384 , N=16384 , K=16384 , allreduce_size = 134217728*4 , datatype = "half"  , bias = False)

  csv_file.close()


def nccl_conv2d_tests():

  csv_file = os.path.join(args.csv_dir, "conv_allreduce.csv") 
  csv_file = open(csv_file, 'w') 
  csv_writer = csv.writer(csv_file)

  csv_header = ["allreduce_size", "n_in", "c_in", "h_in", "w_in", "c_out", "kernel"] + \
      ["overlap_time", "overlap_op1_time", "overlap_op2_time", "seri_time", "seri_op1_time", "seri_op2_time"]
  csv_writer.writerow(csv_header)

  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024     , n_in=256  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024     , n_in=384  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024     , n_in=512  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024     , n_in=640  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024     , n_in=768  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024     , n_in=896  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024     , n_in=1024 , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)

  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*2   , n_in=256  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*2   , n_in=384  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*2   , n_in=512  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*2   , n_in=640  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*2   , n_in=768  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*2   , n_in=896  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*2   , n_in=1024 , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)

  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*4   , n_in=256  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*4   , n_in=384  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*4   , n_in=512  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*4   , n_in=640  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*4   , n_in=768  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*4   , n_in=896  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*4   , n_in=1024 , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)

  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*8   , n_in=256  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*8   , n_in=384  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*8   , n_in=512  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*8   , n_in=640  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*8   , n_in=768  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*8   , n_in=896  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*8   , n_in=1024 , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)

  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*16  , n_in=256  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*16  , n_in=384  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*16  , n_in=512  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*16  , n_in=640  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*16  , n_in=768  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*16  , n_in=896  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*16  , n_in=1024 , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)

  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*32  , n_in=256  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*32  , n_in=384  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*32  , n_in=512  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*32  , n_in=640  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*32  , n_in=768  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*32  , n_in=896  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*32  , n_in=1024 , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)

  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*64  , n_in=256  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*64  , n_in=384  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*64  , n_in=512  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*64  , n_in=640  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*64  , n_in=768  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*64  , n_in=896  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*64  , n_in=1024 , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)

  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*128 , n_in=256  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*128 , n_in=384  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*128 , n_in=512  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*128 , n_in=640  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*128 , n_in=768  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*128 , n_in=896  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*128 , n_in=1024 , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)

  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*256 , n_in=256  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*256 , n_in=384  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*256 , n_in=512  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*256 , n_in=640  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*256 , n_in=768  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  #nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*256 , n_in=896  , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)
  nccl_conv2d_interference(csv_writer , allreduce_size=1024*1024*256 , n_in=1024 , c_in=3 , h_in=128 , w_in=128 , c_out=8 , kernel=8 , datatype="half" , bias=False)

  csv_file.close()



gem_nccl_tests()
#nccl_conv2d_tests()
#temp_interference()
torch.cuda.synchronize()
