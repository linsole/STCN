"""
model.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""


import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from model.network import STCN
from model.losses import LossComputer, iou_hooks_mo, iou_hooks_so
from util.log_integrator import Integrator
from util.image_saver import pool_pairs


class STCNModel:
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1):
        self.para = para
        self.single_object = para['single_object']
        self.local_rank = local_rank

        self.STCN = nn.parallel.DistributedDataParallel(
            STCN(self.single_object).cuda(), 
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        # Setup logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        if self.single_object:
            self.train_integrator.add_hook(iou_hooks_so)
        else:
            self.train_integrator.add_hook(iou_hooks_mo)
        self.loss_computer = LossComputer(para)

        self.train()
        self.optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.STCN.parameters()), lr=para['lr'], weight_decay=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])
        if para['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.report_interval = 100
        self.save_im_interval = 800
        self.save_model_interval = 50000
        if para['debug']:
            self.report_interval = self.save_im_interval = 1

    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        #: data is a dictionary, the keys are: 'rgb', 'gt', 'cls_gt', (sec_gt and selector),  'info',
        #: while value belongs to Tensors.
        #: the loop mainly focuses on transfering data to GPU
        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        Fs = data['rgb'] #: 4,3,3,384,384
        Ms = data['gt'] #: 4,3,1,384,384

        #: autocast automatically cast the data precision
        with torch.cuda.amp.autocast(enabled=self.para['amp']):
            # key features never change, compute once
            #: key frames are image frames without object masks
            k16, kf16_thin, kf16, kf8, kf4 = self.STCN('encode_key', Fs)
            #: k16: 4,64,3,24,24    kf16_thin: 4,3,512,24,24
            #: kf16: 4,3,1024,24,24    kf8: 4,3,512,48,48    kf4: 4,3,256,96,96

            #: single object happens only during pre-training
            if self.single_object:
                #: encode the value of the reference frame
                ref_v = self.STCN('encode_value', Fs[:,0], kf16[:,0], Ms[:,0])  #: 8,512,1,24,24

                # Segment frame 1 with frame 0
                prev_logits, prev_mask = self.STCN('segment', 
                        k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                        k16[:,:,0:1], ref_v, Ms[:,0]) #: prev_logits: 8,2,384,384    prev_mask:8,1,384,384

                #: previous value is relative to the third frame, i.e. it is the second 
                #: frame's encoded value
                prev_v = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask) #: prev_v: 8,512,1,24,24

                values = torch.cat([ref_v, prev_v], 2) #: values: 8,512,2,24,24

                del ref_v

                # Segment frame 2 with frame 0 and 1
                this_logits, this_mask = self.STCN('segment', 
                        k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                        k16[:,:,0:2], values, prev_mask) #: this_logits: 8,2,384,384    this_mask:8,1,384,384

                out['mask_1'] = prev_mask #: 8,1,384,384
                out['mask_2'] = this_mask #: 8,1,384,384
                out['logits_1'] = prev_logits #: 8,2,384,384
                out['logits_2'] = this_logits #: 8,2,384,384
            else:
                sec_Ms = data['sec_gt'] #: 4,3,1,384,384
                selector = data['selector'] #: 4,2

                #: Why use the information of the other mask when encoding one mask? 
                #: Well, simply because STM does that. Also it uses only 2 object masks to 
                #: encode in order to save computational memory cost. See below github issues:
                #: https://github.com/hkchengrex/STCN/issues/77
                #: https://github.com/hkchengrex/STCN/issues/85
                ref_v1 = self.STCN('encode_value', Fs[:,0], kf16[:,0], Ms[:,0], sec_Ms[:,0]) #: 4,512,1,24,24
                ref_v2 = self.STCN('encode_value', Fs[:,0], kf16[:,0], sec_Ms[:,0], Ms[:,0]) #: 4,512,1,24,24
                ref_v = torch.stack([ref_v1, ref_v2], 1) #: 4,2,512,1,24,24

                # Segment frame 1 with frame 0
                prev_logits, prev_mask = self.STCN('segment', 
                        k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1], 
                        k16[:,:,0:1], ref_v, Ms[:,0], sec_Ms[:,0], selector) #: prev_logits: 4,3,384,384    prev_mask: 4,2,384,384
                
                prev_v1 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,0:1], prev_mask[:,1:2]) #: 4,512,1,24,24
                prev_v2 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,1:2], prev_mask[:,0:1]) #: 4,512,1,24,24
                prev_v = torch.stack([prev_v1, prev_v2], 1) #: 4,2,512,1,24,24
                values = torch.cat([ref_v, prev_v], 3) #: 4,2,512,2,24,24

                del ref_v

                # Segment frame 2 with frame 0 and 1
                this_logits, this_mask = self.STCN('segment', 
                        k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2], 
                        k16[:,:,0:2], values, prev_mask[:,0:1], prev_mask[:,1:2], selector) #: this_logits: 4,3,384,384    this_mask: 4,2,384,384

                out['mask_1'] = prev_mask[:,0:1] #: 4,1,384,384
                out['mask_2'] = this_mask[:,0:1] #: 4,1,384,384
                out['sec_mask_1'] = prev_mask[:,1:2] #: 4,1,384,384
                out['sec_mask_2'] = this_mask[:,1:2] #: 4,1,384,384

                out['logits_1'] = prev_logits #: 4,3,384,384
                out['logits_2'] = this_logits #: 4,3,384,384

            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.save_im_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)
                                self.logger.log_cv2('train/pairs', pool_pairs(images, size, self.single_object), it)

            if self._is_train:
                if (it) % self.report_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                #: save the model after every save_model_interval
                if it % self.save_model_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save(it)

            # Backward pass
            # This should be done outside autocast
            # but I trained it like this and it worked fine
            # so I am keeping it this way for reference
            self.optimizer.zero_grad(set_to_none=True)
            if self.para['amp']:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward() 
                self.optimizer.step()
            self.scheduler.step()

    def save(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        torch.save(self.STCN.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)

        self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = { 
            'it': it,
            'network': self.STCN.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.STCN.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    nn.init.orthogonal_(pads)
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.STCN.module.load_state_dict(src_dict)
        print('Network weight loaded:', path)

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # Shall be in eval() mode to freeze BN parameters
        self.STCN.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.STCN.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.STCN.eval()
        return self

