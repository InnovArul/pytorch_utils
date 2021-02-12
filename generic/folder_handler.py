"""Image folder handler with HDF5 support
"""
import h5py, io, glob
import os, os.path as osp, sys
from natsort import natsorted
from PIL import Image
import numpy as np, filetype
from tqdm import tqdm
from icecream import ic
from pathlib import Path

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

class BaseFolder:
    """system folder constructor

    Args:
        root ([str]): root folder path
    """
    def __init__(self, root):
        self.root = root

    def glob(self, path, pattern):
        path_list = self.get_folder_contents(path, return_paths=True)
        return [p for p in path_list if Path(p).match(pattern)]

class SystemFolder(BaseFolder):
    """folder handler
    """
    def __init__(self, root):
        """system folder constructor

        Args:
            root ([str]): root folder path
        """
        super().__init__(root=root)
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

    def exists(self):
        return osp.exists(self.root)

    def glob(self, path, pattern):
        # path_list = self.get_folder_contents(path, return_paths=True)
        # return [p for p in path_list if Path(p).match(pattern)]
        return glob.glob(pattern)

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


class HDF5Folder(BaseFolder):
    """HDF5 folder handler
    """
    def __init__(self, root, h5_file):
        super().__init__(root=root)
        self.h5_file = h5_file
        self.folder_content_memoize = {}

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
        h5_key = f"/LOGISTICS/{osp.basename(self.root)}" + h5_key # + "/__NAMES__"
        if h5_key in self.folder_content_memoize: return self.folder_content_memoize[h5_key]

        with HDF5_Content(self.h5_file, h5_key) as group:
            files = natsorted(list(group.get('__NAMES__')))

        if return_paths:
            files = [osp.join(path, file) for file in files]

        self.folder_content_memoize[h5_key] = files
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
        h5_key = f"/FILES/{osp.basename(self.root)}" + h5_key

        with HDF5_Content(h5_file, h5_key) as obj:
            byte_arr = np.array(obj)

        img = Image.open(io.BytesIO(byte_arr))
        return img

    def exists(self):
        return True

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
    h5_file = "/media/data1/arul/github/unsupervised_personreid/data/dukemtmc-vidreid/DukeMTMC-VideoReID/gallery.hdf5"

    sys_folder = GetFolder(root=folder)
    h5_folder = GetFolder(root=folder, h5_file=h5_file)

    assert sys_folder.get_folder_contents() == h5_folder.get_folder_contents()
    all_class_folders = sys_folder.get_folder_contents()
    for subf in tqdm(all_class_folders):
        curr_folder = osp.join(folder, subf)
        verify_recursive(curr_folder)
