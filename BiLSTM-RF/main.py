from eval_Human.eval_Human import eval_Human_AC, eval_Human_CT
import sys
import argparse

from eval_Testsets.eval_testsets import eval_testsets
from eval_Yeast.eval_Yeast import eval_Yeastcore_AC, eval_Yeastcore_CT


def parameter_parser():
    parser = argparse.ArgumentParser(description="SAE, reproduce by thnhan@hueuni.edu.vn.")

    parser.add_argument("--human-dset",
                        nargs="?",
                        default=sys.path[0] + '/data/Human',
                        help="Human dataset")

    parser.add_argument("--human-size",
                        nargs="?",
                        default=(3899, 4262),
                        help="Size of Human dataset")

    parser.add_argument("--yeastcore-dset",
                        nargs="?",
                        default=sys.path[0] + '/data/Yeastcore',
                        help="Yeastcore datasets")

    parser.add_argument("--testset",
                        nargs="?",
                        default=sys.path[0] + '/data/Testsets',
                        help="Test datasets")

    parser.add_argument("--yeastcore-size",
                        nargs="?",
                        default=(5594, 5594),
                        help="Size of Yeastcore dataset")

    parser.add_argument("--fixlen",
                        type=int,
                        default=500,
                        help="Fixed-legnth of protein sequences. Default is 500.")

    parser.add_argument("--feature_dim",
                        type=int,
                        default=3605,
                        help="Dimension")

    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="Number of training epochs. Default is 50.")

    parser.add_argument("--batch",
                        type=int,
                        default=128,
                        help="Size of batch. Default is 128.")

    parser.add_argument("--validation",
                        type=int,
                        default=5,
                        help="5 cross-validation.")

    return parser.parse_args()


args = parameter_parser()

# eval_Human_AC(args)
# eval_Human_CT(args)
# eval_Yeastcore_AC(args)
# eval_Yeastcore_CT(args)
# eval_testsets(args)
