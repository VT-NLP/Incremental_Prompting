import argparse
import os
import glob


def define_arguments(parser):
    parser.add_argument('--json-root', type=str, default="./data", help="")
    parser.add_argument('--stream-file', type=str, default="data/MAVEN/streams.json", help="")
    parser.add_argument('--batch-size', type=int, default=1, help="")
    parser.add_argument('--init-slots', type=int, choices=[34, 10], default=34, help="")
    parser.add_argument('--patience', type=int, default=3, help="")
    parser.add_argument('--grad-accumulate-step', type=int, default=8, help="")
    parser.add_argument('--input-dim', type=int, default=2048, help="")
    parser.add_argument('--hidden-dim', type=int, default=512, help="")
    parser.add_argument('--max-slots', type=int, choices=[169, 34], default=169, help="")
    parser.add_argument('--prompt-size', type=int, choices=[169, 34], default=169, help="")
    parser.add_argument('--nhead', type=int, default=8, help="")
    parser.add_argument('--nlayers', type=int, default=3, help="")
    parser.add_argument('--no-gpu', action="store_true", help="don't use gpu")
    parser.add_argument('--gpu', type=int, default=0, help="gpu")
    parser.add_argument('--learning-rate', type=float, default=1e-4, help="")
    parser.add_argument('--decay', type=float, default=1e-2, help="")
    parser.add_argument('--tau', type=float, default=0.5, help="")
    parser.add_argument('--seed', type=int, default=2147483647, help="random seed")
    parser.add_argument('--perm-id', type=int, default=0, help="")
    parser.add_argument('--save-model', type=str, default="emp", help="checkpoints name")
    parser.add_argument('--load-model', type=str, default="emp", help="path to saved checkpoint")
    parser.add_argument('--log-dir', type=str, default="./logs/emp/", help="path to save log file")
    parser.add_argument('--model-dir', type=str, default="./logs/emp/emp.0", help="path to load model checkpoint")
    parser.add_argument('--train-epoch', type=int, default=10, help='epochs to train')
    parser.add_argument('--period', type=int, default=10, help='exemplar replay interval')
    parser.add_argument('--eloss_w', type=int, default=50, help='exemplar replay loss weight')
    parser.add_argument('--test-only', action="store_true", help='is testing')
    parser.add_argument('--finetune', action="store_true", help='')
    parser.add_argument('--load-first', type=str, default="", help="path to saved checkpoint")
    parser.add_argument('--balance', choices=[ 'none', 'fd', 'mul', 'nod'], default="none")
    parser.add_argument('--setting', choices=['classic'], default="classic")

def parse_arguments():
    parser = argparse.ArgumentParser()
    define_arguments(parser)
    args = parser.parse_args()
    args.log = os.path.join(args.log_dir, "logfile.log")
    if (not args.test_only) and os.path.exists(args.log_dir):
        existing_logs = glob.glob(os.path.join(args.log_dir, "*"))
        for _t in existing_logs:
            os.remove(_t)
    return args
