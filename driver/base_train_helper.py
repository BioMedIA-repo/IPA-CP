# -*- coding: utf-8 -*-
# @Time    : 19/11/8 9:54
# @Software: PyCharm
from commons.utils import *
from tensorboardX import SummaryWriter
import torch
import matplotlib.pyplot as plt
from torch.cuda import empty_cache
from models.EMA import MeanTeacher
from models import net_factory_3d
from commons.ramps import sigmoid_rampup
import torch.optim as optim
from medpy import metric

plt.rcParams.update({'figure.max_open_warning': 20})


class BaseTrainHelper(object):
    def __init__(self, criterions, config):
        self.criterions = criterions
        self.config = config
        self.use_cuda = config.use_cuda
        self.device = config.gpu if self.use_cuda else None
        if self.config.train:
            self.make_dirs()
        self.define_log()
        self.reset_model()
        self.out_put_summary()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor

    def make_dirs(self):
        if not isdir(self.config.tmp_dir):
            os.makedirs(self.config.tmp_dir)
        if not isdir(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        if not isdir(self.config.save_model_path):
            os.makedirs(self.config.save_model_path)
        if not isdir(self.config.tensorboard_dir):
            os.makedirs(self.config.tensorboard_dir)
        if not isdir(self.config.submission_dir):
            os.makedirs(self.config.submission_dir)
        code_path = join(self.config.submission_dir, 'code')
        if os.path.exists(code_path):
            shutil.rmtree(code_path)
        print(os.getcwd())
        shutil.copytree('../../', code_path, ignore=shutil.ignore_patterns('.git', '__pycache__', '*log*', '*tmp*'))

    def define_log(self):
        now = datetime.now()  # current date and time
        date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
        if self.config.train:
            log_s = self.config.log_file[:self.config.log_file.rfind('.txt')]
            self.log = Logger(log_s + '_' + str(date_time) + '.txt')
        else:
            self.log = Logger(join(self.config.save_dir, 'test_log_%s.txt' % (str(date_time))))
        sys.stdout = self.log

    def create_model(self):
        model = net_factory_3d(net_type=self.config.model, in_chns=self.config.channels, class_num=self.config.classes)
        return model

    def reset_model(self):
        if hasattr(self, 'model'):
            del self.model
            empty_cache()
        print("Creating models....")
        self.model = self.create_model()
        self.model.to(self.device)

    def count_parameters(self, net):
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def create_mtc(self, decay=0.999, model=None):
        if hasattr(self, 'ema'):
            del self.ema
        mtc = self.create_model()
        mtc = mtc.to(self.device)
        if model is not None:
            mtc.load_state_dict(model.state_dict())
        for param in mtc.parameters():
            param.detach_()
        self.ema = MeanTeacher(mtc, decay)

    def move_to_cuda(self):
        if self.use_cuda and self.model:
            torch.cuda.set_device(self.config.gpu)
            self.equipment = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.equipment)
            for key in self.criterions.keys():
                print(key)
                self.criterions[key].to(self.equipment)
            if len(self.config.gpu_count) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.config.gpu_count)
        else:
            self.equipment = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_iter_checkpoint(self, fold=0, iter_num=0):
        save_model_path = join(self.config.save_model_path, "fold_%s_iter_%s.pt" % (str(fold), str(iter_num)))
        ema_model_path = join(self.config.save_model_path, "fold_%s_iter_%s_ema.pt" % (str(fold), str(iter_num)))
        torch.save(self.model.state_dict(), save_model_path)
        if hasattr(self, 'ema'):
            torch.save(self.ema, ema_model_path)

    def load_iter_checkpoint(self, fold=0, iter_num=0):
        save_model_path = join(self.config.save_model_path, "fold_%s_iter_%s.pt" % (str(fold), str(iter_num)))
        print('loaded:' + save_model_path)
        state_dict_file = torch.load(save_model_path, map_location=('cuda:' + str(self.device)))
        net_dict = self.model.state_dict()
        net_dict.update(state_dict_file)
        self.model.load_state_dict(net_dict)
        del state_dict_file
        del net_dict
        if self.config.ema_decay > 0:
            if hasattr(self, 'ema'):
                del self.ema
            ema_model_path = join(self.config.save_model_path, "fold_%s_iter_%s_ema.pt" % (str(fold), str(iter_num)))
            print('load ema:' + ema_model_path)
            self.ema = torch.load(ema_model_path, map_location=('cuda:' + str(self.device)))

    def save_best_checkpoint(self, fold=0):
        save_model_path = join(self.config.save_model_path, "fold_%s_best_model.pt" % (str(fold)))
        ema_model_path = join(self.config.save_model_path, "fold_%s_best_ema.pt" % (str(fold)))
        torch.save(self.model.state_dict(), save_model_path)
        if hasattr(self, 'ema'):
            torch.save(self.ema, ema_model_path)

    def load_best_state(self, fold=0):
        save_model_path = join(self.config.save_model_path, "fold_%s_best_model.pt" % (str(fold)))
        print('load best: ' + save_model_path)
        state_dict_file = torch.load(save_model_path, map_location=('cuda:' + str(self.device)))
        net_dict = self.model.state_dict()
        net_dict.update(state_dict_file)
        self.model.load_state_dict(net_dict)
        del state_dict_file
        del net_dict
        if hasattr(self, 'ema'):
            del self.ema
            ema_model_path = join(self.config.save_model_path, "fold_%s_best_ema.pt" % (str(fold)))
            print('load best ema: ' + ema_model_path)
            self.ema = torch.load(ema_model_path, map_location=('cuda:' + str(self.device)))


    def adjust_learning_rate_g(self, optimizer, i_iter, num_steps):
        warmup_iter = num_steps // 20
        if i_iter < warmup_iter:
            lr = self.lr_warmup(self.config.learning_rate, i_iter, warmup_iter)
        else:
            lr = self.lr_poly(self.config.learning_rate, i_iter, num_steps, 0.9)
        # lr = lr if lr > self.config.min_lrate else self.config.min_lrate
        for g in optimizer.param_groups:
            g['lr'] = lr
        return lr

    def lr_warmup(self, base_lr, iter, warmup_iter):
        return base_lr * (float(iter) / warmup_iter)

    def lr_poly(self, base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    def reset_optim(self):
        if self.config.learning_algorithm == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate,
                                  momentum=0.9, weight_decay=0.0001)
            print("Reset optimize parameters: ", self.config.learning_algorithm)
        elif self.config.learning_algorithm == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
            print("Reset optimize parameters: ", self.config.learning_algorithm)
        else:
            raise ValueError("Optim %s not implemented." % (self.config.learning_algorithm))
        return optimizer

    def out_put_summary(self):
        self.summary_writer = SummaryWriter(self.config.tensorboard_dir)
        print('Model has param %.2fM' % (self.count_parameters(self.model) / 1000000.0))

    def write_summary(self, iter_num, criterions, type='val'):
        for key in criterions.keys():
            self.summary_writer.add_scalar(
                type + '/' + key, criterions[key], iter_num)

    def get_current_consistency_weight(self, iter_num, consistency, consistency_rampup):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return consistency * sigmoid_rampup(iter_num, consistency_rampup)

    def cal_metric(self, gt, pred):
        if pred.sum() > 0 and gt.sum() > 0:
            dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            return np.array([dice, hd95])
        else:
            return np.zeros(2)
