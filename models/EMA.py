# -*- coding: utf-8 -*-
# @Time    : 20/10/9 16:42
# @Software: PyCharm
# @Desc    : EMA.py
# class EMA():
#     def __init__(self, model, decay):
#         self.model = model
#         self.decay = decay
#         self.shadow = {}
#         self.backup = {}
#         self.register()
#
#     def register(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()
#
#     def update(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
#                 self.shadow[name] = new_average.clone()
#
#     def apply_shadow(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.shadow
#                 self.backup[name] = param.data
#                 param.data = self.shadow[name]
#
#     def restore(self):
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 assert name in self.backup
#                 param.data = self.backup[name]
#         self.backup = {}
#
class MeanTeacher(object):
    def __init__(self, model, alpha, buffer_ema=True):
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]
        print("Create EMA with decay %.4f" % (self.alpha))

    def update(self, model):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name]
                + (1 - decay) * state[name]
            )
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(
                    decay * self.shadow[name]
                    + (1 - decay) * state[name]
                )
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1
        self.apply_shadow()

    def apply_shadow(self):
        self.model.load_state_dict(self.shadow)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }


class EMA(object):
    '''
        apply expontential moving average to a model. This should have same function as the `tf.train.ExponentialMovingAverage` of tensorflow.
        usage:
            model = resnet()
            model.train()
            ema = EMA(model, 0.9999)
            ....
            for img, lb in dataloader:
                loss = ...
                loss.backward()
                optim.step()
                ema.update_params() # apply ema
            evaluate(model)  # evaluate with original model as usual
            ema.apply_shadow() # copy ema status to the model
            evaluate(model) # evaluate the model with ema paramters
            ema.restore() # resume the model parameters
        args:
            - model: the model that ema is applied
            - alpha: each parameter p should be computed as p_hat = alpha * p + (1. - alpha) * p_hat
            - buffer_ema: whether the model buffers should be computed with ema method or just get kept
        methods:
            - update_params(): apply ema to the model, usually call after the optimizer.step() is called
            - apply_shadow(): copy the ema processed parameters to the model
            - restore(): restore the original model parameters, this would cancel the operation of apply_shadow()
    '''

    def __init__(self, model, alpha, buffer_ema=True):
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]
        print("Create EMA with decay %.4f" % (self.alpha))

    def update(self):
        decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        state = self.model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(
                decay * self.shadow[name]
                + (1 - decay) * state[name]
            )
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(
                    decay * self.shadow[name]
                    + (1 - decay) * state[name]
                )
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }
