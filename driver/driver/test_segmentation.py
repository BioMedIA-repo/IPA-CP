# -*- coding: utf-8 -*-
# @Time    : 21/7/20 10:46
# @Software: PyCharm
# @Desc    : test_segmentation.py
import matplotlib
import sys

sys.path.extend(["../../", "../", "./"])
from driver.FistulaHelper import *
from driver.driver import HELPER
from driver.Config import Configurable

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse


def main(config, args):
    criterion = {
    }

    seg_help = HELPER[config.data_name](criterion,
                                        config)
    seg_help.move_to_cuda()
    print("data name ", seg_help.config.data_name)
    print("data size ", (seg_help.config.patch_x, seg_help.config.patch_y, seg_help.config.patch_z))
    nb_acl_iter = 0
    seg_help.log.flush()
    if seg_help.config.ema_decay > 0:
        seg_help.create_mtc(decay=seg_help.config.ema_decay)

    test(seg_help, fold=nb_acl_iter, args=args)
    seg_help.log.flush()



def test(seg_help, fold, args):
    test_data = seg_help.get_test_data_loader()
    seg_help.load_best_state(fold)
    print("-" * 20, "final seg eval", "-" * 20)
    if hasattr(seg_help, 'ema'):
        seg_help.ema.model.eval()
        seg_help.model.eval()
        print("-" * 20, "tea final seg eval", "-" * 20)
        seg_help.predict_whole(seg_help.ema.model, test_data, vis=True, isTest=True)
        print("-" * 20, "stu final seg eval", "-" * 20)
        seg_help.predict_whole(seg_help.model, test_data, vis=True, isTest=True)
    else:
        seg_help.model.eval()
        print("-" * 20, "stu final seg eval", "-" * 20)
        seg_help.predict_whole(seg_help.model, test_data, vis=True, isTest=True)
    seg_help.log.flush()


if __name__ == '__main__':
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    # torch.backends.cudnn.benchmark = True  # cudn
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config-file', default='seg_configuration.txt')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--train', help='test not need write', default=False)
    argparser.add_argument('--gpu-count', help='number of GPUs (0,1,2,3)', default='0')
    argparser.add_argument('--iter-num', help='number of iters', default=0, type=int)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args, extra_args)
    torch.set_num_threads(config.workers + 1)

    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config, args=args)
