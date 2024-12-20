import itertools
import os.path

import tifffile
import torch
from collections import Counter,OrderedDict
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm
import numpy as np
from itertools import product

from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel
from basicsr.archs import build_network

@MODEL_REGISTRY.register()
class RCAN3DModel(VideoBaseModel):

    def __init__(self, opt):
        super(RCAN3DModel, self).__init__(opt)
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')
            self.midlqloss = opt['train'].get('midlqloss')
            self.clean_fix = opt['train'].get('clean_fix')
    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)
        if self.clean_fix:
            logger = get_root_logger()
            if current_iter/50000 % 2 == 1:
                logger.info(f'Clean Blocks Fix.')
                for name, param in self.net_g.named_parameters():
                    if 'clean' in name:
                        param.requires_grad_(False)
            elif current_iter/50000 % 2 == 0:
                logger.info(f'Clean Blocks Unfix.')
                for name, param in self.net_g.named_parameters():
                    if 'clean' in name:
                        param.requires_grad_(True)
        self.optimizer_g.zero_grad()
        self.output= self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            if self.midlqloss:
                l_pix = self.cri_pix(self.output, self.midlq)
            else:
                l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # clean loss
        if self.cri_clean:
            l_clean = self.cri_clean(self.midlq, self.nf)
            l_total += l_clean
            loss_dict['l_clean'] = l_clean

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        num_folders = len(dataset)
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        for i in range(num_folders):

            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            img_name = osp.splitext(osp.basename(val_data['lq_path']))[0]
            # folder = val_data['folder']

            # compute outputs
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)

            self.net_g.eval()
            input_shape = eval(self.opt['val']['input_shape'])
            overlap_shape = eval(self.opt['val']['overlap_shape'])
            self.output= apply(self.net_g,self.lq,input_shape=input_shape,overlap_shape=overlap_shape)

            self.net_g.train()
            visuals = self.get_current_visuals_nontensor()
            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.gt
            torch.cuda.empty_cache()

            if i < num_folders:

                result = visuals['result']
                result_img = result  # uint8, bgr
                # if 'midlq' in visuals:
                #     midlq = visuals['midlq'][0, idx, :, :, :]
                #     midlq_img = tensor2img([midlq])
                metric_data['img'] = result_img
                if 'gt' in visuals:
                    gt = visuals['gt']
                    gt_img = gt.numpy()  # uint8, bgr
                    metric_data['img2'] = gt_img

                if save_img:
                    save_img_path = osp.join(self.opt['path']['visualization'],dataset_name,
                                             f'{img_name}.tif')
                    os.makedirs(osp.join(self.opt['path']['visualization'],dataset_name),exist_ok=True)
                    tifffile.imwrite(save_img_path,result_img)
                # calculate metrics
                if with_metrics:
                    # calculate metrics
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (i + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def test(self):
        self.net_g.eval()

        with torch.no_grad():
            self.output= self.net_g(self.lq)

        self.net_g.train()

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

def apply(model, data, overlap_shape,input_shape):
    # Get model's input and output shapes
    model_input_image_shape = input_shape
    model_output_image_shape = input_shape

    # Scale ratio between output and input
    scale_factor = [o / i for i, o in zip(model_input_image_shape, model_output_image_shape)]
    def _scale_tuple(t):
        return tuple(int(v * f) for v, f in zip(t, scale_factor))

    # Setup default overlap if not given
    if overlap_shape is None:
        overlap_shape = (2, 32, 32) if len(model_input_image_shape) == 3 else (32, 32)

    # Step size when scanning the image
    step_shape = [m - o for m, o in zip(model_input_image_shape, overlap_shape)]

    # Block weight with ramp
    block_weight = _create_block_weight(model_output_image_shape, overlap_shape, scale_factor)
    block_weight = torch.tensor(block_weight).to(device='cuda:0')
    # Convert data into list if not
    input_is_list = isinstance(data, (list, tuple))
    if not input_is_list:
        data = [data]

    result = []
    for image in data:
        # Ensure image has the correct shape
        image_shape = image.shape[-3:]
        assert len(image_shape) == len(model_input_image_shape), "Error in image shape"

        # Image to tensor
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0) # Batch size: 1
        result_image = np.zeros((1, *image_shape), dtype=np.float32)
        weight_image = np.zeros(image_shape, dtype=np.float32)

        # Determine the number of steps in each dimension
        num_steps = [i // s + (i % s != 0) for i, s in zip(image_shape, step_shape)]

        # Iterate blocks
        for block_idx in tqdm(list(product(*[range(ns) for ns in num_steps]))):
            tl = [s * b for s, b in zip(step_shape, block_idx)] # top left corner
            br = [min(t + m, i) for t, m, i in zip(tl, model_input_image_shape, image_shape)] # bottom right corner

            slice_in = tuple(slice(s, e) for s, e in zip(tl, br))
            slice_out = tuple(slice(0, e-s) for s, e in zip(tl, br))
            # Pad image if necessary
            input_block = image_tensor[:,:,slice_in[0],slice_in[1],slice_in[2]]
            if input_block.shape[-len(model_input_image_shape):] != model_input_image_shape:
                pad_size = [(m - s, 0) for m, s in zip(model_input_image_shape, input_block.shape[-len(model_input_image_shape):])]
                pad_size = [(0, 0)] * (len(input_block.shape) - len(pad_size)) + pad_size
                pad_size = tuple(itertools.chain(*pad_size))[::-1]
                input_block = torch.nn.functional.pad(input_block, pad_size)
            # Predict and weight
            with torch.no_grad():
                pred = model(input_block)
            pred *= block_weight
            pred_np = pred.squeeze(0).cpu().numpy()
            # Write results back to images
            # print(result_image[0][0, :, :],weight_image[0, :, :])
            result_image[:,slice_in[0],slice_in[1],slice_in[2]] += pred_np[:,slice_out[0],slice_out[1],slice_out[2]]
            weight_image[slice_in] += block_weight[slice_out].cpu().numpy()

        # Normalize
        result_image /= weight_image[ np.newaxis,...]
        result_image[np.isnan(result_image)] = 0
        result.append(result_image)

    return result if input_is_list else result[0]

def _create_block_weight(output_shape, overlap_shape, scale_factor):
    block_weight = np.ones([int(m-2*o*f) for m, o, f in zip(output_shape, overlap_shape, scale_factor)], dtype=np.float32)
    block_weight = np.pad(block_weight, [(int(o*f), int(o*f)) for o, f in zip(overlap_shape, scale_factor)], mode='linear_ramp')
    return block_weight

