import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop, three_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class NF3DDataset(data.Dataset):

    def __init__(self, opt):
        super(NF3DDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root, self.nf_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq']), Path(opt['dataroot_nf'])
        self.num_frame = opt['num_frame']
        if 'intensity_threshold' in opt:
            self.intensity_threshold = opt['intensity_threshold']
        if 'area_threshold' in opt:
            self.area_threshold = opt['area_threshold']

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:08d}/{frame_num}' for i in range(int(frame_num))])

        # self.frame_num = int(frame_num)

        # # remove the video clips used in validation
        # if opt['val_partition'] == 'REDS4':
        #     val_partition = ['000', '011', '015', '020']
        # elif opt['val_partition'] == 'official':
        #     val_partition = [f'{v:03d}' for v in range(240, 270)]
        # else:
        #     raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
        #                      f"Supported ones are ['official', 'REDS4'].")
        # if opt['test_mode']:
        #     pass
        #     # self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        # else:
        #     pass
        #     # self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name,frame_num = key.split('/')  # key example: 000/00000000

        self.frame_num = int(frame_num)
        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > self.frame_num - self.num_frame * interval:
            start_frame_idx = random.randint(0, self.frame_num - self.num_frame * interval)
        end_frame_idx = start_frame_idx + self.num_frame * interval
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        img_nfs = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:08d}'
                img_gt_path = f'{clip_name}/{neighbor:08d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:08d}.png'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:08d}.png'
                img_nf_path = self.nf_root / clip_name / f'{neighbor:08d}.png'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes,flag='unchanged', float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes,flag='unchanged', float32=True)
            img_gts.append(img_gt)

            # get NF
            img_bytes = self.file_client.get(img_nf_path, 'nf')
            img_nf = imfrombytes(img_bytes,flag='unchanged', float32=True)
            img_nfs.append(img_nf)

        # randomly crop
        img_gts, img_lqs , img_nfs= three_random_crop(img_gts, img_lqs, img_nfs,gt_size, scale, img_gt_path)
        img_lqs_np = np.array(img_lqs)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_lqs.extend(img_nfs)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 3:2*len(img_lqs) // 3], dim=0)
        img_nfs = torch.stack(img_results[(2*len(img_lqs) // 3):], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 3], dim=0)

        img_gts = torch.transpose(img_gts , 0, 1)
        img_lqs = torch.transpose(img_lqs, 0, 1)
        img_nfs = torch.transpose(img_nfs, 0, 1)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        if np.count_nonzero(img_lqs_np > self.intensity_threshold) < (np.prod(img_lqs_np.shape) *self.area_threshold):
            return {'lq': img_lqs, 'gt': img_gts, 'nf': img_nfs,'key': key,'valid':False}
        else:
            return {'lq': img_lqs, 'gt': img_gts, 'nf': img_nfs,'key': key,'valid':True}

    def __len__(self):
        return len(self.keys)

