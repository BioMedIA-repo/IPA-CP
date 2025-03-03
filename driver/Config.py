from configparser import ConfigParser
import configparser
import sys, os

sys.path.append('..')


class Configurable(object):
    def __init__(self, args, extra_args):
        config = ConfigParser()
        config._interpolation = configparser.ExtendedInterpolation()
        config.read(args.config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        self._config = config
        if args.train:
            if args.gpu != -1:
                config.set('Run', 'gpu', args.gpu)
            if args.run_num != "None":
                config.set('Run', 'run_num', args.run_num)
            if args.model != "None":
                config.set('Network', 'model', args.model)
            if args.default_size != "None":
                config.set('Run', 'default_seg_label_size', args.default_size)
            config.set('Network', 'ema_decay', args.ema_decay)
            config.set('Run', 'train_seg_batch_size', args.bs)
            config.set('Run', 'n_epochs', args.epochs)
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            config.write(open(self.config_file, 'w'))
        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    def set_attr(self, section, name, value):
        return self._config.set(section, name, value)

    # ------------data config reader--------------------
    @property
    def patch_x(self):
        return self._config.getint('Data', 'patch_x')

    @property
    def patch_y(self):
        return self._config.getint('Data', 'patch_y')

    @property
    def patch_z(self):
        return self._config.getint('Data', 'patch_z')

    @property
    def data_name(self):
        return self._config.get('Data', 'data_name')

    @property
    def data_path(self):
        return self._config.get('Data', 'data_path')

    @property
    def data_path(self):
        return self._config.get('Data', 'data_path')

    @property
    def csv_data_path(self):
        return self._config.get('Data', 'csv_data_path')

    # ------------save path config reader--------------------

    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def config_file(self):
        return self._config.get('Save', 'config_file')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')

    @property
    def tmp_dir(self):
        return self._config.get('Save', 'tmp_dir')

    @property
    def submission_dir(self):
        return self._config.get('Save', 'submission_dir')

    @property
    def tensorboard_dir(self):
        return self._config.get('Save', 'tensorboard_dir')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def log_file(self):
        return self._config.get('Save', 'log_file')

    # ------------Network path config reader--------------------

    @property
    def model(self):
        return self._config.get('Network', 'model')

    @property
    def classes(self):
        return self._config.getint('Network', 'classes')

    @property
    def channels(self):
        return self._config.getint('Network', 'channels')

    @property
    def backbone(self):
        return self._config.get('Network', 'backbone')

    @property
    def ema_decay(self):
        return self._config.getfloat('Network', 'ema_decay')

    # ------------Network path config reader--------------------

    @property
    def epochs(self):
        return self._config.getint('Run', 'N_epochs')

    @property
    def test_batch_size(self):
        return self._config.getint('Run', 'test_batch_size')

    @property
    def gpu(self):
        return self._config.getint('Run', 'gpu')

    @property
    def printfreq(self):
        return self._config.getint('Run', 'printfreq')

    @property
    def gpu_count(self):
        gpus = self._config.get('Run', 'gpu_count')
        gpus = gpus.split(',')
        return [int(x) for x in gpus]

    @property
    def patience(self):
        return self._config.getint('Run', 'patience')

    @property
    def workers(self):
        return self._config.getint('Run', 'workers')

    @property
    def run_num(self):
        return self._config.getint('Run', 'run_num')

    @property
    def train_seg_batch_size(self):
        return self._config.getint('Run', 'train_seg_batch_size')

    @property
    def test_seg_batch_size(self):
        return self._config.getint('Run', 'test_seg_batch_size')

    @property
    def default_seg_label_size(self):
        return self._config.getint('Run', 'default_seg_label_size')

    # ------------Optimizer path config reader--------------------
    @property
    def learning_algorithm(self):
        return self._config.get('Optimizer', 'learning_algorithm')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def min_lrate(self):
        return self._config.getfloat('Optimizer', 'min_lrate')

    @property
    def epsilon(self):
        return self._config.getfloat('Optimizer', 'epsilon')

    @property
    def clip(self):
        return self._config.getfloat('Optimizer', 'clip')

    @property
    def update_every(self):
        return self._config.getint('Run', 'update_every')
