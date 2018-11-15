import os
import glob   
import re
import pprint
import argparse


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

parser.add_argument('--logfile', type=str, default="./logfile_rank*.txt", help='logging output')

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

path = args.logfile  
parsed_path = '/home/scratch.shawnw_gpu/docker/nccl-tests/results/parsed/'   
files=glob.glob(path)

total_rank = len(files)
#initial_star_time = {} 
bucket_reduce_time = {}

table = {}
for file in files:
    filename = os.path.basename(file)
    file_name_split = filename.split('_')
    print(file_name_split)
    rank_id = int(file_name_split[2])
    source_file = open(file, "r")

    iteration_idx = 0
    bucket_reduce_time[rank_id] = []

    for line in source_file:
      new_line = line.rstrip()
      line_split = new_line.split(';')
      print(line_split)

      if 'barrier' in line:
        initial_star_time = float(line_split[1])
      elif 'bucket_reduce' in line:
        bucket_reduce_time[rank_id].append(['reduce',iteration_idx, line_split[1], float(line_split[2])- initial_star_time])
      elif 'Iteration_forward_time' in line:
        iteration_idx += 1
        bucket_reduce_time[rank_id].append(['forward_end',line_split[1], '', float(line_split[2])-initial_star_time])
        assert (iteration_idx == int(line_split[1])), "iteration number mis-match"

pprint.pprint(bucket_reduce_time)

# ('forward_end', ' 7', 10.462412595748901),
#       ('reduce', 7, '0', 11.019340515136719),
#       ('reduce', 7, '1', 11.019686222076416),
#       ('reduce', 7, '2', 11.020018339157104),
#       ('reduce', 7, '3', 11.020348310470581),
#       ('reduce', 7, '4', 11.020678043365479),
#       ('reduce', 7, '5', 11.021007776260376),
#       ('reduce', 7, '6', 11.021311283111572),

for rank_id in range(total_rank):
  if rank_id > 0:
    for i in range(len(bucket_reduce_time[rank_id])):
      bucket_reduce_time[0][i].append(bucket_reduce_time[rank_id][i][3])

print("===================================================")
pprint.pprint(bucket_reduce_time)

#write_file = open(os.path.join(parsed_path, "parsed.csv"),"w")
#
#column_list = []
#column_list.append(byte_list)
#name_list = []
#for cname in table.keys():
#    name_list.append(cname)
#    column_list.append(table[cname])
#
#zip_list = zip(*column_list)
#
#header = "bytes;" + ";".join(name_list)
#write_file.write(header+"\n")
#
#for item in zip_list:
#    write_file.write(";".join(item)+"\n")
#
#write_file.close()
