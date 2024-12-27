import cv2
import numpy as np
import os
import sys
import tifffile
import math
import shutil


input_path_list = [
                   r"path\input\Aberrated",
                   r"path\input\GT",
                   r"path\input\NF",
                   r"path\input\Validation\Aberrated",
                   r"path\input\Validation\GT",
                    r"path\input\Validation\NF",
                   ]

out_path_list = [
                 r'path\output\train\Abe',
                 r'path\output\train\GT',
                 r'path\output\train\NF',
                 r'path\output\val\Abe',
                 r'path\output\val\GT',
                 r'path\output\val\NF']

os.makedirs(out_path_list[0], exist_ok=True)
meta_info_txt = r'path\output\meta_info_SynObj.txt'


def normalize(image, p_min=2, p_max=99.9, dtype='float32'):
    '''
    Normalizes the image intensity so that the `p_min`-th and the `p_max`-th
    percentiles are converted to 0 and 1 respectively.

    References
    ----------
    Content-Aware Image Restoration: Pushing the Limits of Fluorescence
    Microscopy
    https://doi.org/10.1038/s41592-018-0216-7
    '''
    low, high = np.percentile(image, (p_min, p_max))
    return (image - low) / (high - low + 1e-6).astype(dtype)

def ceil_xy(img_x):
    if img_x < 256:
        img_x1 = 256
    else:
        img_x1 = math.ceil(img_x / 8) * 8
    return img_x1

def pad_xy(img_x,img_y):
    pad_x = ceil_xy(img_x) - img_x
    pad_y = ceil_xy(img_y) - img_y
    return pad_x,pad_y


with open(meta_info_txt, 'w') as f:
    for j in range(3):
        inpath = input_path_list[j]
        out_path = out_path_list[j]
        img_list = os.listdir(inpath)
        for i in range(len(img_list)):
            tif_name = os.path.splitext(img_list[i])[0]
            img_name = img_list[i]
            img_path = os.path.join(inpath, img_name)
            img_pick = tifffile.imread(img_path)
            img_pick = cv2.normalize(img_pick, None, 0, 255, cv2.NORM_MINMAX)
            img_pick = img_pick.astype('uint8')
            out_folder_path = os.path.join(out_path, tif_name)
            os.makedirs(out_folder_path, exist_ok=True)
            for k in range(img_pick.shape[0]):
                img_out_name = '%08d' % k + '.png'
                img_out = img_pick[k]
                img_x = img_out.shape[0]
                img_y = img_out.shape[1]
                pad_x,pad_y = pad_xy(img_x,img_y)
                img_out = np.pad(img_out,((pad_x//2,pad_x-pad_x//2),(pad_y//2,pad_y-pad_y//2)),"reflect")
                out_img_path = os.path.join(out_folder_path, img_out_name)
                tifffile.imwrite(out_img_path, img_out)
            if j==0:
                info = f'{tif_name}' + f' %d' % img_pick.shape[0] + f' ({img_pick.shape[1]},{img_pick.shape[2]},{1})'
                if i == len(img_list) - 1:
                    f.write(f'{info}')
                else:
                    f.write(f'{info}\n')
    for j in range(3):
        inpath = input_path_list[j+3]
        out_path = out_path_list[j+3]
        shutil.copytree(inpath, out_path)
