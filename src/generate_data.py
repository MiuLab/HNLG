import argparse
import numpy.random as random
import os
from utils import print_time_info

parser = argparse.ArgumentParser(
        description='Process the data and parameters.')
parser.add_argument(
        '--data_dir', help='the directory of data')
parser.add_argument(
        '--min_length', type=int, default=5,
        help='the min length of generated sequence [5]')
parser.add_argument(
        '--max_length', type=int, default=20,
        help='the max length of generated sequence [20]')
parser.add_argument(
        '--data_size', type=int, default=25000,
        help='the generated data size [25000]')
parser.add_argument(
        '--vocab_size', type=int, default=10,
        help='the vocab size of sequences [10]')
parser.add_argument(
        '--reverse', type=int, default=0,
        help='reverse the output sequence or not [0]')
args = parser.parse_args()

print_time_info("Data size: {}".format(args.data_size))
print_time_info("Min length: {}".format(args.min_length))
print_time_info("Max length: {}".format(args.max_length))
print_time_info("Vocab size: {}".format(args.vocab_size))
print_time_info("Start generate data...")

lengths = random.randint(args.min_length, args.max_length+1, args.data_size)
data = [random.randint(0, args.vocab_size, length) for length in lengths]
labels = [d[::-1] if args.reverse else d for d in data]
with open(os.path.join(args.data_dir, "data.txt"), 'w') as file:
    for idx in range(args.data_size):
        d_string = ' '.join(map(str, data[idx]))
        l_string = ' '.join(map(str, labels[idx]))
        file.write("{} | {}\n".format(d_string, l_string))

print_time_info("Done")
