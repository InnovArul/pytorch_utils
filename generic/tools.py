from __future__ import division, print_function, absolute_import
import os
import sys
import json
import time
import errno
import numpy as np
import random
import os.path as osp
import warnings
import PIL
import torch
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import zipfile

__all__ = [
    'mkdir_if_missing', 'check_isfile', 'read_json', 'write_json',
    'set_random_seed', 'download_url', 'read_image', 'collect_env_info',
    'parse_path', 'show_image', 'unzip_file'
]


def show_image(image):
    dpi = 80
    figsize = (image.shape[1] / float(dpi), image.shape[0] / float(dpi))
    fig = plt.figure(figsize=figsize)
    plt.imshow(image)
    fig.show()


def parse_path(path):
    path = osp.abspath(path)
    parent, fullfilename = osp.split(path)
    filename, ext = osp.splitext(fullfilename)
    return parent, filename, ext

def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib
    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    with urllib.request.urlopen(url) as response, open(dst, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    
    sys.stdout.write('\n')


def unzip_file(filepath, destdir):
    """unzip the given zip file to destination directory.

    Args:
        filepath (str): zip file path to be extracted
        destpath (str): source path to extract the zip file contents to
    """
    print('unzipping {} to {}'.format(filepath, destdir))
    with zipfile.ZipFile(filepath,"r") as zip_ref:
        zip_ref.extractall(destdir)


def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img


def collect_env_info():
    """Returns env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info
    env_str = get_pretty_env_info()
    env_str += '\n        Pillow ({})'.format(PIL.__version__)
    return env_str


def get_current_time(f='l'):
    """get current time
    :param f: 'l' for log, 'f' for file name
    :return: formatted time
    """
    if f == 'l':
        return time.strftime('%m/%d %H:%M:%S', time.localtime(time.time()))
    elif f == 'f':
        return time.strftime('%m_%d_%H_%M', time.localtime(time.time()))


def save_scripts(path, scripts_to_save=None):
    """To backup files (typically, before starting an experiment) """
    if not os.path.exists(os.path.join(path, 'scripts')):
        os.makedirs(os.path.join(path, 'scripts'))

    if scripts_to_save is not None:
        for script in scripts_to_save:
            dst_path = os.path.join(path, 'scripts', script)
            try:
                shutil.copy(script, dst_path)
            except IOError:
                os.makedirs(os.path.dirname(dst_path))
                shutil.copy(script, dst_path)


def load_image_in_PIL(path, mode='RGB'):
    """Read image as PIL """
    img = Image.open(path)
    img.load()  # Very important for loading large image
    return img.convert(mode)


def print_cuda_mem(info=None):
    """To print cuda memory usage. """
    if info:
        print(info, end=' ')
    mem_allocated = round(torch.cuda.memory_allocated() / 1048576)
    mem_cached = round(torch.cuda.memory_cached() / 1048576)
    print(f'Mem allocated: {mem_allocated}MB, Mem cached: {mem_cached}MB')


def set_bn_eval(m):
    """To set batchnorm in the model to eval mode

    usage: model.apply(set_bn_eval)
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
