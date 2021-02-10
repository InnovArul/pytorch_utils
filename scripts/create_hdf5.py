import h5py
import numpy as np
import os, os.path as osp
import argparse
from tqdm import tqdm

def create_hdf5_from_dataset(dataset_dir, result_hdf5):
    """create hdf5 file from a dataset folder with images

    Args:
        dataset_dir (str): dataset folder with images
        result_hdf5 (str): hdf5 path
    """    
    hf = h5py.File(save_path, 'w')  # open the file in write mode

    for i in os.listdir(base_path):   # read all the internal dirs
        split_dir = os.path.join(base_path, i)
        print("digesting ", split_dir)
        grp = hf.create_group(i)  # create a hdf5 group

        for class_name in tqdm(os.listdir(split_dir)):  # read all classnames
            class_dir = os.path.join(split_dir, class_name)
            subgrp = grp.create_group(class_name)  # create a subgroup for the above created group.

            for imgname in os.listdir(class_dir):   # find all images inside class dir
                img_path = os.path.join(class_dir, imgname)
                with open(img_path, 'rb') as img_f:  # open images as python binary
                    binary_data = img_f.read()

                binary_data_np = np.asarray(binary_data)
                dset = subgrp.create_dataset(imgname, data=binary_data_np) # save it in the subgroup. each a-subgroup contains all the images.

    hf.close()

def verify_hdf5(dataset_dir, result_hdf5):
    """to verify the hdf5 file contents with the dataset dir with images

    Args:
        dataset_dir (str): dataset dir with images
        result_hdf5 (str): hdf5 file to be verified
    """
    print(f"verify ({dataset_dir, result_hdf5})")
    hf = h5py.File(save_path, 'r')  # open the file in read mode
    for i in os.listdir(base_path):   # read all the internal dirs
        split_dir = os.path.join(base_path, i)
        print(f"verifying {split_dir}")
        grp = hf.get(i)  # get the hdf5 group

        for class_name in tqdm(os.listdir(split_dir)):  # read all classnames
            class_dir = os.path.join(split_dir, class_name)
            subgrp = grp.get(class_name)  # get the correct subgroup

            for imgname in os.listdir(class_dir):   # find all images inside class dir
                img_path = os.path.join(class_dir, imgname)
                with open(img_path, 'rb') as img_f:  # open images as python binary
                    binary_data = img_f.read()

                binary_data_dir = np.asarray(binary_data)
                binary_data_hdf5 = np.array(subgrp.get(imgname)) 

                # verify data from dir and hdf5 are same
                assert np.allclose(binary_data_dir, binary_data_hdf5), f"inconsistent {img_path}"

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
