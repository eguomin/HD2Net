import os
from os import path as osp
from PIL import Image

from basicsr.utils import scandir


def generate_meta_info_div2k():
    """Generate meta info for DIV2K dataset.
    """

    gt_folder = r'F:\DeAbe\Data\TP_Video_X4\train\GT'
    meta_info_txt = r'F:\DeAbe\Data\TP_Video_X4\train\GT\meta_info_TP_Video.txt'

    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')

def generate_meta_info_reds():
    """Generate meta info for REDS dataset.
    """

    gt_folder = r''
    meta_info_txt = r''

    with open(meta_info_txt, 'w') as f:
        for i in range(84):
            info = '%03d' % i + f' 40 ({512},{512},{1})'
            print(i + 1, info)
            f.write(f'{info}\n')

def generate_meta_info_SO_Video():
    """Generate meta info for SO_Video dataset.
    """

    gt_folder = r''
    meta_info_txt = r''

    with open(meta_info_txt, 'w') as f:
        for i in range(1000):
            info = '%03d' % i + f' 5 ({512},{512},{1})'
            print(i + 1, info)
            f.write(f'{info}\n')

def generate_meta_info_TP_Video():
    """Generate meta info for SO_Video dataset.
    """

    abe_folder = r''
    meta_info_txt = r''

    with open(meta_info_txt, 'w') as f:
        dir_list = os.listdir(abe_folder)
        for i in range(len(dir_list)):
            img_list = os.listdir(os.path.join(abe_folder,dir_list[i]))
            info = '%03d' % i + f' %d ' % len(img_list) + f'(512,512,1)'
            f.write(f'{info}\n')

if __name__ == '__main__':
    generate_meta_info_TP_Video()
