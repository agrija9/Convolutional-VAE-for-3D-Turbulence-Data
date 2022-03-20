import torch
# import torch.nn as nn
# from torchvision.utils import save_image, make_grid
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.decomposition import TruncatedSVD

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def init_weights(model):
    """
    Set weight initialization for Conv3D in network.
    Based on: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/24
    """
    if isinstance(model, torch.nn.Conv3d):
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.constant_(model.bias, 0)
        # torch.nn.init.zeros_(model.bias)

def plot_cube_slice(cube_sample, z_slice, channel):
    """
    Plots a horizontal slice of a cube in the z direction.
    """
    plt.imshow(cube_sample[:,:,z_slice,channel]) # [sample_idx] (x,y,z,channel)
    plt.colorbar()
    plt.plot()

def show_histogram(values, norm_func):
    """
    """
    print(values.shape)
    n, bins, patches = plt.hist(values.reshape(-1), 50, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(norm_func(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.viridis(c))

    plt.show()

def plot_cube(cube, IMG_DIM, norm_func, angle=320):
    """
    """
    # right now it works better if normalize again here
    cube = norm_func(cube)

    # apply heatmap, the object viridis is a callable,
    # that when passed a float between 0 and 1 returns an RGBA value from the colormap
    facecolors = cm.viridis(cube)

    # the filled array tells matplotlib what to fill (any true value is filled)
    filled = facecolors[:,:,:,-1] != 0
    x, y, z = np.indices(np.array(filled.shape) + 1)

    # define 3D plotting
    fig = plt.figure(figsize=(30/2.54, 30/2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=IMG_DIM+2)
    ax.set_ylim(top=IMG_DIM+2)
    ax.set_zlim(top=IMG_DIM+2)

    # ax.scatter(x, y, z, filled, facecolors=facecolors, shade=False)
    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    plt.show()

def plot_reconstruction_grid(original, reconstruction, channels, cube_dim, epoch, save_grid=False, reference_batch=False):
    """
    Plots the 3 channels (velocity components) of the original and reconstruction
    cubes in a grid.

    original: torch tensor torch.Size([128, 3, 21, 21, 21])
    reconstruction: torch tensor torch tensor torch.Size([128, 3, 21, 21, 21])
    """

    # move: cuda tensor --> cpu tensor --> numpy array
    original = original.cpu().detach().numpy()
    reconstruction = reconstruction.cpu().detach().numpy()

    batch_indices = [0, 1, 2, 3] # plot grid for different batch samples
    # idx = 0 # choose one sample (cube) from 128 batch to plot

    for index in batch_indices:
        original_sample = original[index]
        reconstruction_sample = reconstruction[index]

        # swap axes from torch (3, 21, 21, 21) to numpy (21, 21, 21, 3)
        original_sample = np.transpose(original_sample, (3, 1, 2, 0))
        reconstruction_sample = np.transpose(reconstruction_sample, (3, 1, 2, 0))

        # locate reconstruction-original cubes in dictionary
        cube_samples = {"Orig" : original_sample,
                        "Recon" : reconstruction_sample}

        # dictionary for velocity channels subtitles
        velocity_channels = {0 : "U0", 1 : "U1", 2 : "U2"}

        # set figure dimensions
        fig = plt.figure(figsize=(40, 40))

        subplot_count = 1

        for key, cube_sample in cube_samples.items():
            for channel in range(channels):
                # define subplots
                ax = fig.add_subplot(2, 3, subplot_count, projection="3d")

                # generate heat map for each channel
                facecolors = cm.viridis(cube_sample[:,:,:,channel])

                # the filled array tells matplotlib what to fill (any true value is filled)
                filled = facecolors[:,:,:,-1] != 0
                x, y, z = np.indices(np.array(filled.shape) + 1)

                # define 3D plotting
                ax = fig.gca(projection="3d")
                ax.view_init(elev=30, azim=320)
                ax.set_xlim(right=cube_dim + 2)
                ax.set_ylim(top=cube_dim + 2)
                ax.set_zlim(top=cube_dim + 2)
                ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
                subplot_count += 1

                # set titles, axis fontsizes
                ax.set_title(key + " " + "vel component " + velocity_channels[channel], fontsize=55)
                ax.tick_params(axis="both", which='major', labelsize=30)

        if save_grid:
            if reference_batch:
                plt.savefig("results/REFERENCE_original_reconstruction_grid_epoch_{}_batchid_{}.png".format(epoch, index))
                plt.close()
            else:
                plt.savefig("results/RANDOM_original_reconstruction_grid_epoch_{}_batchid_{}.png".format(epoch, index))
                plt.close()
        else:
            plt.show()

def plot_generation_grid(model, device, grid_size=9, save_grid=False):
    """
    """
    with torch.no_grad():
        samples = torch.randn(grid_size, 27648).to(device)
        # samples = torch.randn(32, 128).to(device)
        samples = model.decode(samples).cpu() # returns a torch.Size([9, 3, 21, 21, 21])
        print("samples decoded", samples.size())

        # grid = make_grid(sample)
        # writer.add_image('sampling', grid, epoch)
        # save_image(sample.view(64, 4, 41, 41), "results/samples_" + str(epoch) + ".png")

# def save_representations(latent_batch, epoch, writer, args):
#     """
#     """
#     # print("latent batch size:", latent_batch.size())
#     nrow = min(latent_batch.size(0), 8)
#     grid = make_grid(latent_batch.view(args.batch_size, 1, 8, 8)[:nrow].cpu(), nrow=nrow, normalize=True)
#     writer.add_image("latent representations", grid, epoch)
#     save_image(grid.cpu(), "results/representations_" + str(epoch) + ".png", nrow=nrow)
#
# def save_projected_representations(latent_batch, epoch, writer, args, download=True):
#     """
#     Projects latent vectors in 2D using PCA and t-SNE.
#     """
#     print("latent batch size:", latent_batch.size()) # (128, 64)
#     np_latent_batch = latent_batch.cpu().numpy()
#     print("np latent batch size:", np_latent_batch.shape)
#
#     PCA_latent_batch = TruncatedSVD(n_components=3).fit_transform(np_latent_batch)
#     tSNE_latent_batch = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=3000).fit_transform(np_latent_batch)
#     print("PCA latent batch shape:", PCA_latent_batch.shape)
#
#     # plot PCA
#     plt.rcParams.update({'font.size': 22})
#     fig = plt.figure(figsize=(12,12))
#     ax1 = fig.add_subplot(111)
#     ax1.scatter(PCA_latent_batch[:,0], PCA_latent_batch[:,1])
#     plt.title('PCA on latent represenations', fontdict = {'fontsize' : 30})
#     plt.xlabel("Principal Component 1", fontsize=22)
#     plt.ylabel("Principal Component 2", fontsize=22)
#     plt.legend()
#     plt.grid()
#
#     if download:
#         plt.savefig("results/projected_representations_" + str(epoch) + "_pca.png")
#     else:
#         plt.show()
#
#     # plot t-SNE
#     plt.rcParams.update({'font.size': 22})
#     fig = plt.figure(figsize=(12,12))
#     ax1 = fig.add_subplot(111)
#     ax1.scatter(tSNE_latent_batch[:,0], tSNE_latent_batch[:,1])
#     plt.title('t-SNE on latent represenations', fontdict = {'fontsize' : 30})
#     plt.legend()
#     plt.grid()
#
#     if download:
#         plt.savefig("results/projected_representations_" + str(epoch) + "_tsne.png")
#     else:
#         plt.show()
