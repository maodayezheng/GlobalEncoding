import sys
from ModelBuilder import debug_test
from ModelBuilder import training

sys.setrecursionlimit(5000000)

main_dir = sys.argv[1]
out_dir = sys.argv[2]

if __name__ == '__main__':
    debug_test("/gpu:0")

