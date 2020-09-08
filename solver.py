
import os
import sys
import datetime
import cv2
import torch
import pprint
import numpy as np
import time
import argparse
import pickle

import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import scipy.io as sio # to save detection as mat files
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from munch import Munch, munchify

from data import BaseTransformTubeLet
from data import dataset_factory
from data import DatasetBuilder
from lib.modeling.layers.modules import MultiBoxLoss
from lib.modeling.layers.box_utils import decode
from lib.utils.evaluation import evaluate_frameAP
from lib.utils.AverageMeter import  AverageMeter
from lib.utils.augmentations_tubelets import SSDAugmentation
from lib.utils.config_parse import cfg
from lib.utils.config_parse import cfg_from_file
from lib.modeling.build_model import build_ssd

from lib.utils.validation_utils import save_tubelets_nms3d
from lib.utils.validation_utils import tubeletNMS
from lib.utils.validation_utils import storeTubeletDetections
from lib.utils.validation_utils import get_frameGT
from lib.utils.validation_utils import final_frameNMS

from lib.utils.visualize_utils import flattenDets
from lib.utils.visualize_utils import plotWithBoxesLabelsAndScores

from eval_utils import Evaluator

import wandb
from tqdm import tqdm

# TODO: Workaround for this global declaration
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class Solver(object):

    def __init__(self, args):
        self.cfg = cfg

        ## Load dataloaders for train and inference
        self.train_loader = self.load_dataset(cfg.DATASET, 'train') if 'train' in cfg.PHASE else None
        self.eval_loader = self.load_dataset(cfg.DATASET, 'eval') if 'eval' in cfg.PHASE else None
        self.test_loader = self.load_dataset(cfg.DATASET, 'test') if ('test' in cfg.PHASE or 'eval_dropout' in cfg.PHASE) else None
        self.visualize_loader = self.load_dataset(cfg.DATASET, 'test') if 'visualize' in cfg.PHASE else None

        ## Create model and shift to GPU
        if self.train_loader is not None:
            cfg.MODEL.NUM_CLASSES = len(self.train_loader.dataset.CLASSES) + 1
        elif self.eval_loader is not None:
            cfg.MODEL.NUM_CLASSES = len(self.eval_loader.dataset.CLASSES) + 1
        elif self.test_loader is not None:
            cfg.MODEL.NUM_CLASSES = len(self.test_loader.dataset.CLASSES) + 1

        self.model, self.priorbox = self.build_model(self.cfg.MODEL)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.model.priors = self.priors
        self.model.priorbox = self.priorbox
        self.init_weights()
        #
        self.cuda = torch.cuda.is_available()
        self.multi_gpu = args.multi_gpu
        if self.cuda:
            print('Utilize GPUs for computation')
            print('Number of GPU available', torch.cuda.device_count())
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            # TODO: CUDNN_INTERNAL_STATUS ERROR WITH I3D Networks
            # cudnn.benchmark = True
            if torch.cuda.device_count() > 1 and self.multi_gpu:
                self.model = torch.nn.DataParallel(self.model)
                self.model.to(device)
        

        wandb.init(project="action-detection", tags=[args.dataset, cfg.MODEL.SSDS, args.input_type, cfg.MODEL.NETS])
        wandb.config.update(self.cfg)

    def load_dataset(self, dataset_cfg, phase='train'):
        dataset = DatasetBuilder(dataset_cfg.DATASET)

        AnnotationTransform = dataset_factory.AnnotationTransform
        detection_collate_tubelet = dataset_factory.detection_collate_tubelet

        ssd_dim = self.cfg.MODEL.IMAGE_SIZE[0]
        means = self.cfg.MODEL.MEANS
        stds = self.cfg.MODEL.STDS
        input_type = dataset_cfg.INPUT_TYPE
        K = self.cfg.MODEL.K
        interval = self.cfg.DATASET.INTERVAL

        kwargsDataset = {
            'cls': dataset, 
            'root': dataset_cfg.DATASET_DIR, 
            'image_set': dataset_cfg.TRAIN_SETS[0], 
            'transform': SSDAugmentation(ssd_dim, means, stds),
            'target_transform' : AnnotationTransform(),
            'input_type': input_type,
            'full_test': False, 
            'num_K': K,
            'split': dataset_cfg.TRAIN_SETS[1], 
            'interval':interval
        }
        
        kwargsDataLoader = {
            'batch_size': self.cfg.TEST.BATCH_SIZE,
            'num_workers': 1,
            'shuffle': False,
            'collate_fn':detection_collate_tubelet,
            'pin_memory':True
        }

        if phase == 'train':
            kwargsDataset['split'] = dataset_cfg.TRAIN_SETS[1]
            kwargsDataset['image_set'] = dataset_cfg.TRAIN_SETS[0]
            
            kwargsDataLoader['shuffle'] = True
            kwargsDataLoader['batch_size'] = self.cfg.TRAIN.BATCH_SIZE
        elif phase == 'eval':
            kwargsDataset['split'] = dataset_cfg.TEST_SETS[1]
            kwargsDataset['image_set'] = 'val'
            kwargsDataset['transform'] = BaseTransformTubeLet(ssd_dim, means, stds)
        elif phase == 'test':
            kwargsDataset['split'] = dataset_cfg.TEST_SETS[1]
            kwargsDataset['image_set'] = dataset_cfg.TEST_SETS[0]
            kwargsDataset['transform'] = BaseTransformTubeLet(ssd_dim, means, stds)
            kwargsDataset['full_test'] = True

        dataset_ = dataset_factory.Dataset_plus_Torch_Class.fromtorch(**kwargsDataset)
        dataloader_ = data.DataLoader(dataset = dataset_, **kwargsDataLoader)

        return dataloader_

    def build_model(self, model_cfg):

        print("Created Model\n")
        # offset_pool = self.cfg.MODEL.OFFSET_POOL
        # global_feat = self.cfg.MODEL.GLOBAL_FEAT
        # rnn = self.cfg.MODEL.RNN
        # K = self.cfg.MODEL.K
        # self.split_gpus = self.cfg.MODEL.SPLIT_GPUS
        # num_classes = self.cfg.MODEL.NUM_CLASSES # Should be background + 1

        model = build_ssd(model_cfg)
        return model

    def create_output_dir(self):
        dataset = self.cfg.DATASET.DATASET
        split = self.cfg.DATASET.TRAIN_SETS[1]
        input_type = self.cfg.DATASET.INPUT_TYPE
        arch = self.cfg.MODEL.SSDS
        basenet = self.cfg.MODEL.NETS
        lr = self.cfg.TRAIN.OPTIMIZER.LEARNING_RATE

        self.exp_name = '{}-{:02d}-{}-{}-{}-lr-{:05d}'.format(dataset, split, arch, basenet, input_type, int(lr * 100000))

        save_root = self.cfg.EXP_DIR + '/cache/' + self.exp_name + '/'

        if not os.path.isdir(save_root):
            os.makedirs(save_root)

        return save_root

    def init_weights(self):
        def xavier(param):
            init.xavier_uniform(param)

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                if (m.bias is not None):
                    m.bias.data.zero_()

        def zero_init(m):
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                m.bias.data.zero_()

        print('Initializing weights for extra layers and HEADs...')

        if self.cfg.DATASET.INPUT_TYPE == 'fusion':
            self.model.rgb_model.feature_extractor.extras.apply(weights_init)
            self.model.rgb_model.predictor.loc.apply(weights_init)
            self.model.rgb_model.predictor.conf.apply(weights_init)
            self.model.rgb_model.extras.apply(weights_init)

            self.model.flo_model.feature_extractor.extras.apply(weights_init)
            self.model.flo_model.predictor.loc.apply(weights_init)
            self.model.flo_model.predictor.conf.apply(weights_init)
            self.model.flo_model.extras.apply(weights_init)
        else:
            self.model.feature_extractor.extras.apply(weights_init)
            self.model.predictor.loc.apply(weights_init)
            self.model.predictor.conf.apply(weights_init)
            self.model.extras.apply(weights_init)

    # For loading from two different checkpoints for FLO and RGB
    def resume_checkpoint_separate(self, resume_checkpoint, resume_scope):
        self.resume_checkpoint_pretrained(self.model.rgb_model, resume_checkpoint[0], resume_scope)
        self.resume_checkpoint_pretrained(self.model.flo_model, resume_checkpoint[1], resume_scope)

    def resume_checkpoint_pretrained(self, model, resume_checkpoint, resume_scope):
        if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
            return False
        print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
        checkpoint = torch.load(resume_checkpoint)

        ## For loading from legacy checkpoints
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        else:
            checkpoint = checkpoint

        # If trained using dataparallel
        if 'module.' in list(checkpoint.items())[0][0]:
            pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
            checkpoint = pretrained_dict

        if torch.cuda.device_count() > 1 and self.multi_gpu:
            torch_model = model.module
        else:
            torch_model = model

        checkpoint = self.load_legacy_checkpoints(checkpoint, torch_model)

        if resume_scope != '':
            pretrained_dict = {}
            for k, v in list(checkpoint.items()):
                for resume_key in resume_scope.split(','):
                    if resume_key in k:
                        pretrained_dict[k] = v
                        break
            checkpoint = pretrained_dict



        # intersection of model and pre-trained model
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in torch_model.state_dict()}
        checkpoint = torch_model.state_dict()

        # Print out variables not being restored
        unresume_dict = set(checkpoint) - set(pretrained_dict)
        if len(unresume_dict) != 0:
            print("=> UNResume weigths:")
            print(unresume_dict)

        # Update model with intersection and load
        checkpoint.update(pretrained_dict)
        # return torch_model.load_state_dict(checkpoint)

        try:
            return torch_model.load_state_dict(checkpoint)
        except RuntimeError:
            print("Warning! Strict load disabled. Some keys might not be loaded")
            # input("Press Enter to proceed...")
            return torch_model.load_state_dict(checkpoint, strict=False)

    def resume_checkpoint(self, ckpt):
        if ckpt == '' or not os.path.isfile(ckpt):
            print(("=> no checkpoint found at '{}'".format(ckpt)))
            return False
        print(("=> loading checkpoint '{:s}'".format(ckpt)))
        if torch.cuda.device_count() > 1 and self.multi_gpu:
            torch_model = self.model.module
        else:
            torch_model = self.model

        # (08_16_2019): Legacy checkpoint loading trained for single GPU but testing on multi-GPU
        try:
            return torch_model.load_state_dict(torch.load(ckpt))
        except RuntimeError:
            print("Warning! Strict load disabled. Some keys might not be loaded")
            # input("Press Enter to proceed...")
            return self.resume_checkpoint_pretrained(torch_model, ckpt, self.cfg.TRAIN.RESUME_SCOPE)
            # return torch_model.load_state_dict(torch.load(ckpt), strict=False)

    def find_previous(self):
        if not os.path.exists(os.path.join(self.output_dir, 'checkpoint_list.txt')):
            return False
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'r') as f:
            lineList = f.readlines()
        epoches, resume_checkpoints = [list() for _ in range(2)]
        for line in lineList:
            epoch = int(line[line.find('iteration ') + len('iteration '): line.find(':')])
            checkpoint = line[line.find(':') + 2:-1]
            epoches.append(epoch)
            resume_checkpoints.append(checkpoint)
        return epoches, resume_checkpoints

    def save_checkpoint(self, iteration):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename = self.cfg.CHECKPOINTS_PREFIX + '_iter_{:d}'.format(iteration) + '.pth'
        filename = os.path.join(self.output_dir, filename)
        if torch.cuda.device_count() > 1 and self.multi_gpu:
            torch.save(self.model.module.state_dict(), filename)
        else:
            torch.save(self.model.state_dict(), filename)
        with open(os.path.join(self.output_dir, 'checkpoint_list.txt'), 'a') as f:
            f.write('iteration {iteration:d}: {filename}\n'.format(iteration=iteration, filename=filename))

    def set_trainable_param(self, model, trainable_scope, prefix=''):
        for param in model.parameters():
            param.requires_grad = False

        trainable_param = []

        for module in trainable_scope.split(','):
            if torch.cuda.device_count() > 1 and self.multi_gpu:
                dec = hasattr(model.module, module)
                torch_model = model.module
            else:
                dec = hasattr(model, module)
                torch_model = model

            if dec:
                # print(getattr(self.model, module))

                for name, param in getattr(torch_model, module).named_parameters():
                    param.requires_grad = True
                module_params = [(prefix+module+'.'+a[0],a[1])for a in getattr(torch_model, module).named_parameters()]
                trainable_param.extend(module_params)
            else:
                print("Module not found {}".format(module))


        return trainable_param

    def train_loop(self, start_iter=0):
        log_file_path = self.output_dir+"training_{}.log".format(self.cfg.DATASET.DATASET)
        log_file = open(log_file_path, "w" if not os.path.exists(log_file_path) else "a", 1)
        log_file.write(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"))
        log_file.write(pprint.pformat(self.cfg))
        log_file.write(str(self.model))

        # Load parameters for train loop
        eval_step = self.cfg.TRAIN.EVAL_ITER
        max_iter = self.cfg.TRAIN.MAX_ITERATIONS
        print_step = 10
        loss_reset_step = 300

        self.optimizer.zero_grad()
        batch_iterator = None

        # loss counters
        batch_time = AverageMeter()
        losses = AverageMeter()
        loc_losses = AverageMeter()
        cls_losses = AverageMeter()

        epoch_size = len(self.train_loader.dataset)
        t0 = time.perf_counter()

        self.model.train()

        for iteration in range(start_iter, self.cfg.TRAIN.MAX_ITERATIONS + 1):
            if (not batch_iterator) or (iteration % epoch_size == 0):
                batch_iterator = iter(self.train_loader)
            try:
                images, targets, img_indexs = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(self.train_loader)
                images, targets, img_indexs = next(batch_iterator)


            if self.cuda:
                images = Variable(images.cuda())
                with torch.no_grad():
                    targets = [[Variable(anno[0].cuda(0), volatile=True), Variable(anno[1].cuda(0), volatile=True)]
                               for anno in targets]
            else:
                images = Variable(images)
                with torch.no_grad():
                    targets = [[Variable(anno[0], volatile=True), Variable(anno[1], volatile=True)] for anno in targets]
            # forward
            out = self.model(images, phase='train')
            # backprop
            loss_l, loss_c = self.criterion(out, targets)
            loss_l = loss_l
            loss_c = loss_c
            loss = loss_l + loss_c
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            loc_loss = float(loss_l.item())
            conf_loss = float(loss_c.item())

            loc_losses.update(loc_loss)
            cls_losses.update(conf_loss)
            losses.update((loc_loss + conf_loss)/2.0)

            # Dump loss values
            if iteration % print_step == 0 and iteration > 0:
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                batch_time.update(t1 - t0)
                current_lr = self.scheduler.get_lr()[2]

                print_line = 'Itration {:06d}/{:06d} loc-loss {:.3f}({:.3f}) cls-loss {:.3f}({:.3f}) ' \
                             'average-loss {:.3f}({:.3f}) LR {} Timer {:0.3f}({:0.3f})'.format(
                    iteration, max_iter, loc_losses.val, loc_losses.avg, cls_losses.val,
                    cls_losses.avg, losses.val, losses.avg, current_lr, batch_time.val, batch_time.avg)

                self.writer.add_scalar('data/loc_loss', loc_losses.val, iteration)
                self.writer.add_scalar('data/cls_loss', cls_losses.val, iteration)
                self.writer.add_scalar('data/avg_losses', losses.val, iteration)
                self.writer.add_scalar('data/lr', current_lr, iteration)

                wandb.log({"loc_loss": loc_losses.val, 
                           "cls_loss": cls_losses.val,
                           "avg_loss": losses.val,
                           "lr": current_lr},
                           step=iteration)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                log_file.write(print_line + '\n')
                print(print_line)

            # Reset loss queues
            if iteration % loss_reset_step == 0 and iteration > 0:
                loc_losses.reset()
                cls_losses.reset()
                losses.reset()
                batch_time.reset()
                print('Reset accumulators of ', self.exp_name, ' at', iteration)

            # Save step
            if (iteration % eval_step == 0) and iteration > 0:
                torch.cuda.synchronize()
                self.save_checkpoint(iteration)


            # Eval step
            torch.cuda.synchronize()
            tvs = time.perf_counter()
            if (iteration % eval_step == 0) and iteration > 0 and 'eval' in self.cfg.PHASE:
                ap_strs, mAP = self.validation_loop(iteration, phase=['eval'], save=None)
                self.model.train()  # Switch net back to training mode

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                prt_str = '\nValidation TIME::: {:0.3f}\n\n'.format(t0 - tvs)
                print(prt_str)
                log_file.write(prt_str)

                for ap_str in ap_strs:
                    log_file.write(ap_str + '\n')
                log_file.write('Mean AP {}\n'.format(mAP))

                self.writer.add_scalar('data/mAP', mAP, iteration)

                wandb.log({"Val ACC": mAP}, step=iteration)

        # Train Loop End
        log_file.close()

    def validation_loop(self, iteration, phase='eval', save=None):
        self.model.eval()  # switch net to evaluation mode
        with torch.no_grad():
            dataloader_ = self.test_loader if 'test' in phase else self.eval_loader
            mAP, ap_all, ap_strs = self.validate(dataloader_, iteration, phase=phase, save=save)
        return ap_strs, mAP

    def train_model(self):
        # Check if previous models exist and resume those otherwise load pre-trained checkpoint
        self.output_dir = self.create_output_dir()
        self.writer = SummaryWriter(log_dir=self.output_dir)
        # Get Params and set different learning rates for the bias
        if self.cfg.MODEL.INPUT_TYPE == 'fusion':
            parameter_dict = self.set_trainable_param(self.model.rgb_model, self.cfg.TRAIN.TRAINABLE_SCOPE,
                                                      prefix='rgb_model.')
            parameter_dict += self.set_trainable_param(self.model.flo_model, self.cfg.TRAIN.TRAINABLE_SCOPE,
                                                       prefix='flo_model.')
        else:
            parameter_dict = self.set_trainable_param(self.model, self.cfg.TRAIN.TRAINABLE_SCOPE, prefix='')

        params = []
        lr = self.cfg.TRAIN.OPTIMIZER.LEARNING_RATE
        momentum = self.cfg.TRAIN.OPTIMIZER.MOMENTUM
        weight_decay = self.cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY
        gamma = self.cfg.TRAIN.OPTIMIZER.GAMMA

        # Set different learning rate to bias layers and set their weight_decay to 0
        for name, param in parameter_dict:
            if param.requires_grad:
                if name.find('bias') > -1:
                    print(name, 'layer parameters will be trained @ {}'.format(lr * 2))
                    params += [{'params': [param], 'lr': lr * 2, 'weight_decay': 0, 'name': name}]
                elif name.find('offset') > -1:
                    print(name, 'layer parameters will be trained @ {}'.format(lr * 0.1))
                    params += [{'params': [param], 'lr': lr * 0.1, 'weight_decay': weight_decay, 'name': name}]
                else:
                    print(name, 'layer parameters will be trained @ {}'.format(lr))
                    params += [{'params': [param], 'lr': lr, 'weight_decay': weight_decay, 'name': name}]

        self.optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.criterion = MultiBoxLoss(self.cfg.MODEL, self.cfg.MODEL.NUM_CLASSES, 0.5, True, 0, True, 3, 0.5, False, self.cuda)
        if cfg.TRAIN.LR_SCHEDULER.SCHEDULER == 'COSINE':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.TRAIN.MAX_ITERATIONS)
        else:
            self.scheduler = MultiStepLR(self.optimizer, milestones=self.cfg.TRAIN.LR_SCHEDULER.STEPS, gamma=gamma)

        previous = self.find_previous()
        if previous:
            start_iteration = previous[0][-1] + 1
            self.resume_checkpoint(previous[1][-1])
            self.scheduler.step(start_iteration + 1)
        else:
            start_iteration = 0
            if len(self.cfg.RESUME_CHECKPOINT) == 2:
                self.resume_checkpoint_separate(self.cfg.RESUME_CHECKPOINT, self.cfg.TRAIN.RESUME_SCOPE)
            elif len(self.cfg.RESUME_CHECKPOINT) == 1:
                self.resume_checkpoint_pretrained(self.model, self.cfg.RESUME_CHECKPOINT[0], self.cfg.TRAIN.RESUME_SCOPE)

        # Set output directories
        if 'train' in self.cfg.PHASE:
            self.train_loop(start_iteration)

    def test_model(self):
        self.output_dir = self.create_output_dir()

        save_root = os.path.join('results', self.cfg.EXP_DIR)
        save_dir = os.path.join(save_root, self.cfg.DATASET.DATASET)
        self.writer = SummaryWriter(log_dir=self.output_dir)
        previous = self.find_previous()
        iteration = 0

        eval_args = {   
                        'checkpoints_dir':   save_root,              
                        'K':self.cfg.MODEL.K,
                        'dataset':self.cfg.DATASET.DATASET,
                        'path':self.cfg.DATASET.DATASET_DIR,
                        'split':self.test_loader.dataset.split,
                        'redo':False,
                        'eval_mode':self.cfg.DATASET.INPUT_TYPE
                    }

        if previous:
            for iteration, resume_checkpoint in zip(previous[0], previous[1]):
                if self.cfg.TEST.TEST_SCOPE[0] <= iteration <= self.cfg.TEST.TEST_SCOPE[1]:
                    self.resume_checkpoint(resume_checkpoint)
                    self.validation_loop(iteration, phase=self.cfg.PHASE, save=save_dir)

                    ## Call evaluator object
                    eval_args['eval_iter'] = iteration
                    # eval_args['checkpoints_dir'] =  os.path.join(save_dir,
                    #                       self.test_loader.dataset.input_type + '-' + str(self.test_loader.dataset.split).zfill(2) + '-' + str(
                    #                           iteration).zfill(6))
                    evaluator_ = Evaluator(munchify(eval_args))
                    evaluator_.normal_summarize()
                    
        else:
            print("Loading pretrained checkpoint")
            if len(self.cfg.RESUME_CHECKPOINT) == 2:
                self.resume_checkpoint_separate(self.cfg.RESUME_CHECKPOINT, self.cfg.TRAIN.RESUME_SCOPE)
            elif len(self.cfg.RESUME_CHECKPOINT) == 1:
                self.resume_checkpoint_pretrained(self.model, self.cfg.RESUME_CHECKPOINT[0], self.cfg.TRAIN.RESUME_SCOPE)
            self.validation_loop(iteration, phase=self.cfg.PHASE, save=save_dir)

            ## Call evaluator object
            eval_args['eval_iter'] = iteration
            # eval_args['checkpoints_dir'] =  os.path.join(save_dir,
            #                               self.test_loader.dataset.input_type + '-' + str(self.test_loader.dataset.split).zfill(2) + '-' + str(
            #                                   iteration).zfill(6))
            evaluator_ = Evaluator(munchify(eval_args))
            evaluator_.normal_summarize()

    def visualize_model(self):
        self.output_dir = self.create_output_dir()
        dataloader_ = self.visualize_loader

        if dataloader_ is None:
            print("Visualization loader does not exist")
            raise ValueError

        previous = self.find_previous()
        iteration = 0
        if previous:
            for iteration, resume_checkpoint in zip(previous[0], previous[1]):
                if self.cfg.TEST.TEST_SCOPE[0] <= iteration <= self.cfg.TEST.TEST_SCOPE[1]:
                    self.resume_checkpoint(resume_checkpoint)
                    self.visualize(dataloader_, iteration)
                    
        else:
            print("Loading pretrained checkpoint")
            if len(self.cfg.RESUME_CHECKPOINT) == 2:
                self.resume_checkpoint_separate(self.cfg.RESUME_CHECKPOINT, self.cfg.TRAIN.RESUME_SCOPE)
            elif len(self.cfg.RESUME_CHECKPOINT) == 1:
                self.resume_checkpoint_pretrained(self.model, self.cfg.RESUME_CHECKPOINT[0], self.cfg.TRAIN.RESUME_SCOPE)
            self.visualize(dataloader_, iteration)

    def validate(self, val_data_loader, iteration_num, phase='eval', save=None):
        """Test a SSD network on an image database."""
        iou_thresh = self.cfg.TEST.IOU_THRESHOLD
        conf_thresh = self.cfg.POST_PROCESS.SCORE_THRESHOLD
        nms_thresh = self.cfg.POST_PROCESS.IOU_THRESHOLD
        topk = self.cfg.POST_PROCESS.MAX_DETECTIONS


        print('Validating at ', iteration_num)
        val_dataset = val_data_loader.dataset
        num_images = len(val_dataset)
        classes = val_data_loader.dataset.CLASSES
        num_classes = len(classes) + 1


        det_boxes = [{} for _ in range(len(classes))]
        gt_boxes = {} # Video ID and Frame Id should be the key
        print_time = True
        batch_iterator = None
        val_step = 100
        count = 0
        torch.cuda.synchronize()
        ts = time.perf_counter()

        noneCounter = 0

        ## Variables need for saving results
        image_ids = val_dataset.ids
        save_ids = []

        losses = AverageMeter()
        loc_losses = AverageMeter()
        cls_losses = AverageMeter()

        if save:
            output_dir = os.path.join(save,
                                          val_dataset.input_type + '-' + str(val_dataset.split).zfill(2) + '-' + str(
                                              iteration_num).zfill(6))
            log_file = os.path.join(output_dir, 'testing-' + str(iteration_num).zfill(6) + '.log')
            det_file = os.path.join(output_dir, 'detection-' + str(iteration_num).zfill(6) + '.pkl')
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = None

        losses = AverageMeter()
        loc_losses = AverageMeter()
        cls_losses = AverageMeter()

        # for val_itr in range(2):
        for val_itr in tqdm(range(len(val_data_loader)), leave=True):
            if not batch_iterator:
                batch_iterator = iter(val_data_loader)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            images, targets, img_indexs = next(batch_iterator)

            batch_size = images.size(0)
            # height, width = images.size(3), images.size(4)

            if self.cuda:
                images = Variable(images.cuda(), volatile=True)
                with torch.no_grad():
                    targets = [[Variable(anno[0].cuda(0), volatile=True), Variable(anno[1].cuda(0), volatile=True)]
                               for anno in targets]

            output = self.model(images)

            if 'eval' in phase:
                loss_l, loss_c = self.criterion(output, targets)
                loss = loss_l + loss_c
                loc_loss = float(loss_l.item())
                conf_loss = float(loss_c.item())

                loc_losses.update(loc_loss)
                cls_losses.update(conf_loss)
                losses.update((loc_loss + conf_loss)/2.0)

            loc_data = output[0]
            conf_preds = output[1]
            prior_data = output[2]

            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                print('Forward Time {:0.3f}'.format(tf-t1))
            for b in range(batch_size):
                index = img_indexs[b]
                annot_info = image_ids[index]
                frame_range = annot_info[1]
                videoname = val_dataset.video_list[annot_info[0]]
                width, height = val_dataset.get_resolution(videoname)
                save_ids.append([videoname, frame_range])

                # Add gt information
                frame_range_1 = get_frameGT(val_dataset, videoname, frame_range, gt_boxes)

                decoded_boxes = decode(loc_data[b].view(-1,4).data,
                                       prior_data.unsqueeze(1).repeat(1, self.cfg.MODEL.K, 1).view(-1,4).data,
                                       self.priorbox.variance).clone()
                decoded_boxes = decoded_boxes.view(prior_data.size(0), -1, 4)
                conf_scores = self.model.softmax(conf_preds[b]).data.clone()

                # TODO: class agnostic NMS for storing all tubelets - using max score of tubelet
                if save:
                    save_tubelets_nms3d(decoded_boxes, conf_scores, output_dir, videoname, frame_range_1)

                # per-class 3D NMS
                tBoxes, tScores, _ = tubeletNMS(decoded_boxes, conf_scores, width, height, conf_thresh, nms_thresh, topk, num_classes)
                storeTubeletDetections(videoname, frame_range_1, tBoxes, tScores, det_boxes)


                count += 1

            if val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te-ts))
                torch.cuda.synchronize()
                ts = time.perf_counter()
            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                print('NMS stuff Time {:0.3f}'.format(te - tf))

        # TODO: 2D NMS on frame detections -because multiple clips are overwriting to the same frame
        print('Applying 2d NMS', iteration_num)
        final_frameNMS(det_boxes, save, output_dir, nms_thresh, topk)


        print('Evaluating detections for itration number ', iteration_num)
        print('NoneCounter {}'.format(noneCounter))

        mAP, ap_all, ap_strs = evaluate_frameAP(gt_boxes, det_boxes, classes, iou_thresh=iou_thresh)
        aps = [mAP]
        legends = ['mAP']
        ap_to_write = {'mAP': mAP}



        for ap, cls in zip(ap_all, val_data_loader.dataset.CLASSES):
            aps.append(ap)
            legends.append(cls)
            ap_to_write[cls] = ap

        if 'test' in phase:
            with open(log_file, 'w') as f:
                for ap_str in ap_strs:
                    f.write(ap_str + '\n')
                f.write('Mean AP {}\n'.format(mAP))

        for ap_str in ap_strs:
            print(ap_str)
        print('Mean AP {}\n'.format(mAP))

        if save:
            if 'test' in phase:
                with open(det_file, 'wb') as f:
                    pickle.dump([gt_boxes, det_boxes, save_ids], f, pickle.HIGHEST_PROTOCOL)

        if 'eval' in phase:
            wandb.log({"loc_loss_val": loc_losses.avg, 
                "cls_loss_val": cls_losses.avg,
                "avg_loss_val": losses.avg},
                step=iteration_num)

            self.writer.add_scalar('{:s}/loc_loss', loc_losses.avg, iteration_num)
            self.writer.add_scalar('{:s}/cls_loss', cls_losses.avg, iteration_num)
            self.writer.add_scalar('{:s}/avg_losses', losses.avg, iteration_num)
            self.writer.add_scalar('{:s}/mAP', mAP, iteration_num)
        return mAP, ap_all, ap_strs

    def visualize(self, val_data_loader, iteration_num):
        """Test a SSD network on an image database."""
        iou_thresh = self.cfg.TEST.IOU_THRESHOLD
        conf_thresh = self.cfg.POST_PROCESS.SCORE_THRESHOLD
        nms_thresh = self.cfg.POST_PROCESS.IOU_THRESHOLD
        topk = self.cfg.POST_PROCESS.MAX_DETECTIONS


        print('Validating at ', iteration_num)
        val_dataset = val_data_loader.dataset
        num_images = len(val_dataset)
        classes = val_data_loader.dataset.CLASSES
        num_classes = len(classes) + 1


        det_boxes = [{} for _ in range(len(classes))]
        gt_boxes = {} # Video ID and Frame Id should be the key
        batch_iterator = None
        count = 0

        ## Variables need for saving results
        save_root = os.path.join('results', self.cfg.EXP_DIR, self.cfg.DATASET.DATASET)
        image_ids = val_dataset.ids
        save_ids = []

        output_dir = os.path.join(save_root,
                                          val_dataset.input_type + '-' + str(val_dataset.split).zfill(2) + '-' + str(
                                              iteration_num).zfill(6))
        os.makedirs(output_dir, exist_ok=True)

        for val_itr in range(2):
        # for val_itr in tqdm(range(len(val_data_loader)), leave=True):
            if not batch_iterator:
                batch_iterator = iter(val_data_loader)

            images, targets, img_indexs = next(batch_iterator)

            batch_size = images.size(0)
            # height, width = images.size(3), images.size(4)

            if self.cuda:
                with torch.no_grad():
                    images = Variable(images.cuda())
                    targets = [[Variable(anno[0].cuda(0)), Variable(anno[1].cuda(0))]
                               for anno in targets]

            output = self.model(images)

            loc_data = output[0]
            conf_preds = output[1]
            prior_data = output[2]

            for b in range(batch_size):
                index = img_indexs[b]
                annot_info = image_ids[index]
                frame_range = annot_info[1]
                videoname = val_dataset.video_list[annot_info[0]]
                width, height = val_dataset.get_resolution(videoname)
                save_ids.append([videoname, frame_range])

                im_orig = val_dataset.pull_image_tubelet(index)

                # Add gt information
                frame_range_1 = get_frameGT(val_dataset, videoname, frame_range, gt_boxes)

                decoded_boxes = decode(loc_data[b].view(-1,4).data,
                                       prior_data.unsqueeze(1).repeat(1, self.cfg.MODEL.K, 1).view(-1,4).data,
                                       self.priorbox.variance).clone()
                decoded_boxes = decoded_boxes.view(prior_data.size(0), -1, 4)
                conf_scores = self.model.softmax(conf_preds[b]).data.clone()

                # per-class 3D NMS
                tBoxes, tScores, _ = tubeletNMS(decoded_boxes, conf_scores, width, height, conf_thresh, nms_thresh, topk, num_classes)
                dets = flattenDets(tBoxes, tScores, self.cfg.MODEL.K)

                im_to_plot = plotWithBoxesLabelsAndScores(val_dataset, im_orig, dets, threshold=0.7, max_detections=20)
                video_dir = os.path.join(output_dir, videoname)
                output_file_name = video_dir + '/{:s}.jpg'.format('-'.join([str(a).zfill(5) for a in frame_range_1]))
                cv2.imwrite(output_file_name, im_to_plot[0].astype(np.int32))

                count += 1

        print('Evaluating detections for itration number ', iteration_num)