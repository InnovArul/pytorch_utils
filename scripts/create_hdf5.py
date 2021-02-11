import h5py
import numpy as np
import os, os.path as osp
import argparse
from tqdm import tqdm
from PIL import Image
import io, filetype
from natsort import natsorted

def get_names_in_dir(dirname):
    """return file/folder names inside dir in natsorted order

    Args:
        dirname (str): directory path

    Returns:
        str: file/folder names
    """    
    return natsorted(os.listdir(dirname))

def create_hdf5_recursive(hdf5_root, data_dir_or_path):
    # if it is dir, parse through the contents and call 
    # create_hdf5_recursive recursively
    if osp.isdir(data_dir_or_path):
        dirname = osp.basename(data_dir_or_path)
        grp = hdf5_root.create_group(dirname)  # create a hdf5 group

        curr_files_folders = get_names_in_dir(data_dir_or_path)
        for f in curr_files_folders:
            create_hdf5_recursive(grp, osp.join(data_dir_or_path, f))

    # if its an image, read and digest the image
    elif filetype.is_image(data_dir_or_path):
        with open(data_dir_or_path, 'rb') as img_f:  # open images as python binary
            binary_data = img_f.read()

        imgname = osp.basename(data_dir_or_path)
        binary_data_np = np.asarray(binary_data)
        dset = hdf5_root.create_dataset(imgname, data=binary_data_np) # save it in the subgroup. each a-subgroup contains all the images.
    else:
        print(f"{data_dir_or_path} not digestable into hdf5")


def create_hdf5_from_dataset(dataset_dir, result_hdf5):
    """create hdf5 file from a dataset folder with images

    Args:
        dataset_dir (str): dataset folder with images
        result_hdf5 (str): hdf5 path
    """    
    print(f"create_hdf5 ({dataset_dir, result_hdf5})")
    hf = h5py.File(result_hdf5, 'w')  # open the file in write mode
    create_hdf5_recursive(hf, dataset_dir)
    hf.close()

def parse_as_image(byte_arr):
    """reurn the given ninary numpy array as numpy image

    Args:
        byte_arr (np.array): binary np array

    Returns:
        [np.array]: numpy multi-dimensional array image
    """
    return np.array(Image.open(io.BytesIO(byte_arr)))


def verify_hdf5_recursive(hdf5_root, data_dir_or_path):
    # if it is dir, parse through the contents and call 
    # verify_hdf5_recursive recursively
    if osp.isdir(data_dir_or_path):
        dirname = osp.basename(data_dir_or_path)
        grp = hdf5_root.get(dirname)  # get hdf5 group

        curr_files_folders = get_names_in_dir(data_dir_or_path)
        for f in curr_files_folders:
            verify_hdf5_recursive(grp, osp.join(data_dir_or_path, f))

    # if its an image, read and verify the image
    elif filetype.is_image(data_dir_or_path):
        with open(data_dir_or_path, 'rb') as img_f:  # open images as python binary
            binary_data = img_f.read()

        # print(data_dir_or_path)
        imgname = osp.basename(data_dir_or_path)
        binary_data_dir = np.asarray(binary_data)
        binary_data_hdf5 = np.array(hdf5_root.get(imgname))

        # verify data from dir and hdf5 are same
        assert np.allclose(parse_as_image(binary_data_dir), \
                            parse_as_image(binary_data_hdf5)), f"inconsistent {data_dir_or_path}"
        # print("done")

    else:
        print(f"{data_dir_or_path} not digestable into hdf5")


def verify_hdf5(dataset_dir, result_hdf5):
    """to verify the hdf5 file contents with the dataset dir with images

    Args:
        dataset_dir (str): dataset dir with images
        result_hdf5 (str): hdf5 file to be verified
    """
    print(f"verify ({dataset_dir, result_hdf5})")
    hf = h5py.File(result_hdf5, 'r')  # open the file in read mode
    verify_hdf5_recursive(hf, dataset_dir)
    hf.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="images to hdf5")
    parser.add_argument("--dataset-dir", default="", help="dataset root directory")
    parser.add_argument("--result-hdf5", default="", help="path of hdf5 file to be created")
    parser.add_argument("--verify-only", default=False, action="store_true", help="only verify hdf5 with the directory")
    args = parser.parse_args()

    # check if only verification is requested
    if not args.verify_only:
        # create the hdf5 and verify
        create_hdf5_from_dataset(args.dataset_dir, args.result_hdf5)
        verify_hdf5(args.dataset_dir, args.result_hdf5)
    else:
        verify_hdf5(args.dataset_dir, args.result_hdf5)
