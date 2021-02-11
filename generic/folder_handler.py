"""Image folder handler with HDF5 support
"""
import h5py, io
import os, os.path as osp, sys
from natsort import natsorted
from PIL import Image
import numpy as np, filetype
from tqdm import tqdm
from icecream import ic

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

class SystemFolder:
    """folder handler
    """
    def __init__(self, root):
        """system folder constructor

        Args:
            root ([str]): root folder path
        """        
        self.root = root
        assert osp.isdir(root), f"{root} is not a Folder"

    def get_folder_contents(self, path=None, return_paths=False):
        """get the files or folder list inside the given path

        Args:
            path (str): folder path
            return_paths (bool, optional): whether to return as paths. Defaults to False.

        Returns:
            list: folder or file names 
        """        
        if path is None: path = self.root
        assert path.startswith(self.root)

        files = os.listdir(path)
        files = natsorted(files)

        if return_paths:
            files = [osp.join(path, file) for file in files]

        return files

    def read_image(self, path):
        """read image

        Args:
            path (str): image path 

        Returns:
            Image: Image read from given path
        """        
        return Image.open(path)

class HDF5_Content:
    def __init__(self, h5_file, key):
        self.h5_file = h5_file
        self.key = key

    def __enter__(self):
        self.h5_root = h5py.File(self.h5_file, mode="r")
        obj = self.h5_root
        if len(self.key) > 0:
            obj = self.h5_root[self.key]
        return obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5_root.close()

class HDF5Folder:
    """HDF5 folder handler
    """
    def __init__(self, root, h5_file):
        self.root = root
        self.h5_file = h5_file

    def get_folder_contents(self, path=None, return_paths=False):
        """get the files or folder list inside the given path

        Args:
            path (str): folder path
            return_paths (bool, optional): whether to return as paths. Defaults to False.

        Returns:
            list: folder or file names 
        """         
        if path is None: path = self.root
        assert path.startswith(self.root)
        
        h5_key = remove_prefix(path, self.root)
        # h5_key = remove_prefix(h5_key, os.sep)
        h5_key = f"/{osp.basename(self.root)}" + h5_key

        with HDF5_Content(h5_file, h5_key) as group:
            # print(group.keys())
            files = natsorted(list(group.keys()))

        if return_paths:
            files = [osp.join(path, file) for file in files]

        return files

    def read_image(self, path):
        """read image

        Args:
            path (str): image path 

        Returns:
            Image: Image read from given path
        """       
        assert path.startswith(self.root)
        h5_key = remove_prefix(path, self.root)
        # h5_key = remove_prefix(h5_key, os.sep)
        h5_key = f"/{osp.basename(self.root)}" + h5_key

        with HDF5_Content(h5_file, h5_key) as obj:
            byte_arr = np.array(obj)

        img = Image.open(io.BytesIO(byte_arr))
        return img


def GetFolder(root="", h5_file=None):
    if h5_file is not None:
        return HDF5Folder(root=root, h5_file=h5_file)
    else:
        return SystemFolder(root=root)


def verify_recursive(fpath):
    if osp.isdir(fpath):
        fpaths = sys_folder.get_folder_contents(fpath, return_paths=True)
        assert fpaths == h5_folder.get_folder_contents(fpath, return_paths=True)
        for fp in fpaths: verify_recursive(fp)

    elif filetype.is_image(fpath):
        img_sys = sys_folder.read_image(fpath)
        img_h5 = h5_folder.read_image(fpath)
        assert np.allclose(np.array(img_sys), np.array(img_h5))


if __name__ == "__main__":
    folder = "/media/data1/arul/github/unsupervised_personreid/data/dukemtmc-vidreid/DukeMTMC-VideoReID/gallery"
    h5_file = "/media/data1/arul/github/unsupervised_personreid/data/dukemtmc-vidreid/DukeMTMC-VideoReID-HDF5/gallery.hdf5"

    sys_folder = GetFolder(root=folder)
    h5_folder = GetFolder(root=folder, h5_file=h5_file)

    assert sys_folder.get_folder_contents() == h5_folder.get_folder_contents()
    all_class_folders = sys_folder.get_folder_contents()
    for subf in tqdm(all_class_folders):
        curr_folder = osp.join(folder, subf)
        verify_recursive(curr_folder)
