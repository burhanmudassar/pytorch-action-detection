import sys
import argparse
from typing import Union

from solver import Solver
from lib.utils.config_parse import cfg_from_file
from lib.utils.config_parse import cfg
from lib.utils.config_parse import update_cfg

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--cfg', dest='cfg',
            help='experiment config file', default=None, type=str, required=True)
    parser.add_argument('--dataset', dest='dataset', choices=['ucf24','jhmdb','ucfsports','move'],
            help='Choose the dataset', default=None, type=str, required=True)
    parser.add_argument('--dataset_dir', dest='dataset_dir', help='Provide directory of dataset', default=None, type=str, required=True)
    parser.add_argument('--split', dest='split', help='Choose the split of the dataset', default=1, type=int)
    parser.add_argument('--input_type', default='rgb', type=str, help='INput tyep default rgb can take flow as well')
    parser.add_argument('--multi_gpu', dest='multi_gpu', help='Use multi-gpu training or testing', action='store_true')
    parser.add_argument('--step', default='50000,70000,90000', type=str, help='Step iterations for STEP LR')
    parser.add_argument('--max_iter', default=None, type=int, help='Maximum iterations to train for ')
    parser.add_argument('--eval_iter', default=None, type=int, help='Interval for validation')
    parser.add_argument('--init_checkpoint', default=None, help="Specify checkpoints to init from")
    parser.add_argument('--K', dest='K', help='Length of tubelet', default=2, type=int)
    parser.add_argument('--interval', dest='interval', help='Intra-frame interval', default=1, type=int)
    parser.add_argument('--output_dir', dest='output_dir', help='Override default output directory', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg_from_file(args.cfg)

    cfg.DATASET.DATASET = args.dataset
    cfg.DATASET.DATASET_DIR = args.dataset_dir
    cfg.DATASET.TRAIN_SETS = ['train', args.split]
    cfg.DATASET.TEST_SETS = ['test', args.split]
    cfg.DATASET.INPUT_TYPE = args.input_type

    if args.max_iter is not None:
        cfg.TRAIN.MAX_ITERATIONS = args.max_iter

    if args.eval_iter is not None:
        cfg.TRAIN.EVAL_ITER = args.eval_iter

    if args.step is not None:
        cfg.TRAIN.STEPS = [int(itr) for itr in args.step.split(',')]

    if args.init_checkpoint is not None:
        cfg.RESUME_CHECKPOINT = args.init_checkpoint.split(',')

    if args.K is not None:
        cfg.MODEL.K = args.K

    if args.interval is not None:
        cfg.DATASET.INTERVAL = args.interval

    if args.output_dir is not None:
        cfg.EXP_DIR = args.output_dir
        cfg.LOG_DIR = args.output_dir

    update_cfg()

    s = Solver(args)
    s.train_model()
