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
from glob import glob

__all__ = [
    'mkdir_if_missing', 'check_isfile', 'read_json', 'write_json',
    'download_url', 'read_image', 'collect_env_info',
    'parse_path', 'show_image', 'unzip_file', 'load_image_in_PIL',
    'save_scripts', 'get_current_time', 'setup_log_folder'
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
        return time.strftime('%d-%b-%y-%H:%M', time.localtime(time.time()))


def save_scripts(path, scripts_to_save=None):
    """To backup files (typically, before starting an experiment)

     usage:
        myutils.save_scripts(log_dir, scripts_to_save=glob('*.*'))
        myutils.save_scripts(log_dir, scripts_to_save=glob('dataset/*.py', recursive=True))
        myutils.save_scripts(log_dir, scripts_to_save=glob('model/*.py', recursive=True))
        myutils.save_scripts(log_dir, scripts_to_save=glob('myutils/*.py', recursive=True))
    """
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


def setup_log_folder(name, prefix_time=True, log_root='logs', copy_src_files=None):
    """setup log folder

    Args:
        name (str): name of the log folder
        prefix_time (bool, optional): whether to prepend time as prefix. Defaults to True.
        log_root (str, optional): log root folder. Defaults to 'logs'.
        copy_src_files (list or tuple, optional): list of dirs with .py files to be copied. Defaults to None.

    Returns:
        str: log folder path
    """    
    folder_name = name
    # prepend time if needed
    if prefix_time: folder_name = get_current_time(f='f') + '-' + folder_name

    log_folder = osp.join(log_root, folder_name)
    assert not osp.exists(log_folder), f"{log_folder} already exists"
    mkdir_if_missing(log_folder)

    # copy all the source files
    if copy_src_files is not None:
        assert isinstance(copy_src_files, (list, tuple))
        for src_folder in copy_src_files:
            save_scripts(log_folder, scripts_to_save=glob(f'{src_folder}/*.py', recursive=True))

    return log_folder