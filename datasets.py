import numpy as np
import os, sys
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import pdb
from skimage.transform import resize

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class CFD3DDataset(Dataset):
    def __init__(self, data_directory, no_simulations, simulation_timesteps, transforms=None):
        """
        data_directory: path to directory that contains subfolders with the npy files
        Subfolders are folders containing each component of velocity: extract_cubes_U0_reduced
        """

        print()
        print("[INFO] started instantiating 3D CFD pytorch dataset")

        self.data_directory = data_directory
        self.no_simulations = no_simulations # 96
        self.simulation_timesteps = simulation_timesteps # 100
        self.transforms = transforms

        # data_dir = "../cfd_data/HVAC_DUCT/cubes/coords_3d/"
        data_directory_U0 = self.data_directory + "extract_cubes_U0_reduced/"
        data_directory_U1 = self.data_directory + "extract_cubes_U1_reduced/"
        data_directory_U2 = self.data_directory + "extract_cubes_U2_reduced/"

        # read cubes data from directories
        cubes_U0_dict = self._load_3D_cubes(data_directory_U0)
        cubes_U1_dict = self._load_3D_cubes(data_directory_U1)
        cubes_U2_dict = self._load_3D_cubes(data_directory_U2)

        # compare all folders have same simulation parameters
        if self._compare_U_sim_keys(cubes_U0_dict, cubes_U1_dict) and \
           self._compare_U_sim_keys(cubes_U0_dict, cubes_U2_dict) and \
           self._compare_U_sim_keys(cubes_U1_dict, cubes_U2_dict):
            print("[INFO] all folders have same keys (simulations)")
        else:
            print("[INFO] the folders don't have the same keys (simulations)")
            quit()

        # concatenate all velocity components into one dictionary data structure
        cubes_U_all_dict = self._merge_velocity_components_into_dict(cubes_U0_dict, cubes_U1_dict, cubes_U2_dict)

        # creates a list of length timesteps x simulations, each element is a numpy array with cubes size (21,21,21,3)
        # cubes_U_all_channels: 9600 with shape (21,21,21,3)
        self.cubes_U_all_channels = self._concatenate_3_velocity_components(cubes_U_all_dict)
        print("[INFO] cubes dataset length:", len(self.cubes_U_all_channels))
        print("[INFO] single cube shape:", self.cubes_U_all_channels[0].shape)
        self.data_len = len(self.cubes_U_all_channels)

        # stack all cubes in a final numpy array numpy (9600, 21, 21, 21, 3)
        self.stacked_cubes = np.stack(self.cubes_U_all_channels, 0)

        print()
        print("[INFO] mean and std of the cubes dataset along 3 channels")
        # note: not using mean and std separately, just calling them in standardize function (below)
        # note: only standardize data to mean 0 and std 1
        self.mean, self.std = self._compute_mean_std_dataset(self.stacked_cubes)
        print("mean:", self.mean)
        print("std:", self.std)

        # standardize data from here
        print()
        print("[INFO] standardize data to mean 0 and std 1")
        self.standardized_cubes = self._standardize_cubes(self.stacked_cubes)
        print("mean after standardization:", self.standardized_cubes.mean(axis=(0,1,2,3)))
        print("std after standardization:", self.standardized_cubes.std(axis=(0,1,2,3)))

        print()
        print("[INFO] finished instantiating 3D CFD pytorch dataset")

    def _load_3D_cubes(self, data_directory):
        """
        Saves 3D CFD data in a dictionary.
        Keys correspond to .npy file name
        Values correspond to arrays of size (21, 21, 21, 100)
        """

        cubes = {}

        for filename in os.listdir(data_directory):
            if filename.endswith(".npy"):
                # set key without Ui character (for later key matching)
                cubes[filename[2:]] = (np.load(data_directory + "/" + filename))

        return cubes

    def _compare_U_sim_keys(self, cube1, cube2):
        """
        Asserts that two folders with two different velocity componentes
        have same simulation parameters (based on npy file name)
        """
        matched_keys = 0
        for key in cube1:
            if key in cube2:
                matched_keys += 1

        if matched_keys == self.no_simulations:
            return True
        else:
            return False

    def _merge_velocity_components_into_dict(self, cubes_U0, cubes_U1, cubes_U2):
        """
        Concatenates all velocity components U0, U1, U2  based on
        key (simulation name) into a dictionary data structure.
        """
        cubes_U = defaultdict(list)

        for d in (cubes_U0, cubes_U1, cubes_U2): # you can list as many input dicts as you want here
            for key, value in d.items():
                cubes_U[key].append(value)

        # this returns a list of sublists, each sublists contains 3 arrays (corresponding to U0, U1, U2)
        print("[INFO] velocity components concatenated into list")
        return cubes_U

    def _concatenate_3_velocity_components(self, cubes_dict):
        """
        """
        cubes_3_channels = []

        for key, value in cubes_dict.items():
            # split temporal dependency of simulations
            for timestep in range(0, self.simulation_timesteps):
                # fetch velocity compponents
                U0 = cubes_dict[key][0][:,:,:,timestep] # one cube, three channels, one time step
                U1 = cubes_dict[key][1][:,:,:,timestep]
                U2 = cubes_dict[key][2][:,:,:,timestep]

                # concatenate as channels (21, 21, 21, 3)
                U_all_channels = np.concatenate((U0[...,np.newaxis],
                                                 U1[...,np.newaxis],
                                                 U2[...,np.newaxis]),
                                                 axis=3)

                cubes_3_channels.append(U_all_channels)

        return cubes_3_channels

    def _compute_mean_std_dataset(self, data):
        """
        Gets mean and standard deviation values for 3 channels of 3D cube data set.
        It computes mean and standard deviation of full dataset (not on batches)

        Based on: https://stackoverflow.com/questions/47124143/mean-value-of-each-channel-of-several-images
        Based on: https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6

        Returns 1D arrays for mean and std for corresponding channels.
        """

        mean = 0.
        std = 0.

        mean = np.mean(data, axis=(0,1,2,3), dtype=np.float64) # axis=(0,1,2,3)
        std = np.std(data, axis=(0,1,2,3), dtype=np.float64)

        return torch.from_numpy(mean), torch.from_numpy(std)

    def _standardize_cubes(self, data):
        """
        Performs standard normalization on given array.
        """
        # (9600, 21, 21, 21, 3)
        # means = [7.5, 6.3, 1.2]
        return (data - data.mean(axis=(0,1,2,3), keepdims=True)) / data.std(axis=(0,1,2,3), keepdims=True)

    def __getitem__(self, index):
        """
        Returns a tensor cube of shape (3,21,21,21) normalized by
        substracting mean and dividing std of dataset computed beforehand.
        """

        single_cube_numpy = self.standardized_cubes[index] # (21, 21, 21, 3)

        # min-max normalization, clipping and resizing
        single_cube_minmax = self._minmax_normalization(single_cube_numpy) # (custom function)
        single_cube_transformed = np.clip(self._scale_by(np.clip(single_cube_minmax-0.1, 0, 1)**0.4, 2)-0.1, 0, 1) # (from tutorial)
        single_cube_resized = resize(single_cube_transformed, (21, 21, 21), mode='constant') # (21,21,21)

        # swap axes from numpy shape (21, 21, 21, 3) to torch shape (3, 21, 21, 21) this is for input to Conv3D
        # single_cube_reshaped = np.transpose(single_cube_minmax, (3, 1, 2, 0))
        single_cube_reshaped = np.transpose(single_cube_resized, (3, 1, 2, 0))

        # convert cube to torch tensor
        single_cube_tensor = torch.from_numpy(single_cube_reshaped)

        # NOTE: not applying ToTensor() because it only works with 2D images
        # if self.transforms is not None:
            # single_cube_tensor = self.transforms(single_cube_normalized)
            # single_cube_tensor = self.transforms(single_cube_PIL)

        return single_cube_tensor

    def _minmax_normalization(self, data):
       """
       Performs MinMax normalization on given array. Range [0, 1]
       """

       # data shape (21, 21, 21, 3)
       data_min = np.min(data, axis=(0,1,2))
       data_max = np.max(data, axis=(0,1,2))

       return (data-data_min)/(data_max - data_min)

    def _scale_by(self, arr, fac):
        mean = np.mean(arr)
        return (arr-mean)*fac + mean

    def __len__(self):
        return self.data_len
