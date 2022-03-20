import torch
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter
from torchsummary import summary

import argparse
import os
import shutil
import numpy as np
from tqdm import tqdm
import pdb

from models import CVAE_3D, CVAE_3D_II
from train import train
from test import test
from checkpoint import save_checkpoint
from datasets import CFD3DDataset
from utils import init_weights, plot_generation_grid
from loss import schedule_KL_annealing

print()
print("*************")
print("STARTED MAIN")
print("*************")
print()

cuda = torch.cuda.is_available()
if cuda:
    print("[INFO] CUDA available")

device = torch.device("cuda" if cuda else "cpu")
# device = torch.device("cpu")
print("[INFO] device used:", device)

def main():
    """
    Run script: python main.py --test_every_epochs 3 --batch_size 32 --epochs 5 --h_dim 128 --z_dim 64
    """

    parser = argparse.ArgumentParser(description="3D Convolutional Variational Autoencoder")

    # set saving directories an
    parser.add_argument('--result_dir', type=str, default='results', metavar='DIR', help='output directory')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: None')
    parser.add_argument('--test_every_epochs', type=int, default=10, metavar='N', help='test reconstruction, generation every i-th epoch')

    # set model hyperparams and architecture dimensions
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--h_dim', type=int, default=128, metavar='N', help='fully connected hidden units') # DEPRECATED
    parser.add_argument('--z_dim', type=int, default=64, metavar='N', help='latent vector size of encoder')

    args = parser.parse_args()
    # torch.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    # set path to data directory
    dell_path = "cfd_data/HVAC_DUCT/cubes/" # path in Dell/Cluster
    lenovo_path = "cfd_3d_data/cubes/cubes/cubes/" # path in Lenovo

    if os.path.isdir(dell_path):
        data_dir = dell_path # laptop
    else:
        data_dir = "../" + dell_path # cluster

    # create results directory
    try:
        os.makedirs("results/")
    except FileExistsError:
        pass

    # simulation parameters
    no_simulations = 96 # individual npy files
    simulation_timesteps = 100 # time steps per simulation
    IMG_DIM = 21 # cube dimensions
    cube_channels = 3 # 3 velocity components (analogue to RGB)

    # define transforms like cropping, augmentation
    # transformations = transforms.Compose([transforms.CenterCrop(28), transforms.ToTensor()])
    transformations = transforms.Compose([transforms.ToTensor()]) # this is obsolete (not taken into account in datasets.py)

    # define custom 3D dataset
    CFD_3D_dataset = CFD3DDataset(data_dir, no_simulations, simulation_timesteps, transformations)

    # split train, validation sets
    train_set, val_set = torch.utils.data.random_split(CFD_3D_dataset,
                                                       [int(len(CFD_3D_dataset)*0.7),
                                                        int(len(CFD_3D_dataset)*0.3)])

    # create data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True)

    # sample data from loader
    batch_idx, (samples_3D_CFD) = next(enumerate(train_loader)) # this calls __get__item()
    print()
    print("[INFO] data loaded in main, example of 3D cubes batch:", samples_3D_CFD.size()) # (batch, 21, 21, 21, 3)

    # generate reference batch to test reconstruction at every epoch
    reference_batch_3D_CFD = samples_3D_CFD
    # print(reference_batch_3D_CFD[0][2]) # prints 1 cube, 1 channel

    # instantiate model and initialize network weights
    model = CVAE_3D_II(image_channels=cube_channels, h_dim=args.h_dim, z_dim=args.z_dim).to(device=device, dtype=torch.float)
    model.apply(init_weights) # xavier initialization
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # 1e-4 0 KLD, 1e-3 works, 1e-1 & 1e-2 gives NaN
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))

    print()
    print("[INFO] started epoch training")
    start_epoch = 0
    best_test_loss = np.finfo('f').max

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % args.resume)
        else:
            print('=> no checkpoint found at %s' % args.resume)

    writer = SummaryWriter()

    # schedule KL annealing
    kl_weights = schedule_KL_annealing(0.0, 1.0, args.epochs, 4) # cyclical annealing
    kl_weight = 0

    # epoch training
    for epoch in range(start_epoch, args.epochs):
        print()
        print("[INFO] Epoch {}".format(epoch))

        # update KL weight at every epoch
        kl_weight = kl_weights[epoch]
        print("current KL weight:", kl_weight)

        # train losses
        train_total_loss, train_BCE_loss, train_KLD_loss = train(epoch, model, train_loader, kl_weight, optimizer, device, scheduler, args)
        writer.add_scalar("train/train_loss", train_total_loss, epoch) # save loss values with writer (dumped into runs/ dir)
        writer.add_scalar("train/BCE_loss", train_BCE_loss, epoch)
        writer.add_scalar("train/KLD_loss", train_KLD_loss, epoch)
        print("Epoch [%d/%d] train_total_loss: %.3f, train_REC_loss: %.3f, train_KLD_loss: %.3f" % (epoch, args.epochs, train_total_loss, train_BCE_loss, train_KLD_loss))

        # test losses
        if epoch % args.test_every_epochs == 0:
            # test_total_loss, test_BCE_loss, test_KLD_loss = test(epoch, model, test_loader, writer, device, args)
            test_total_loss, test_BCE_loss, test_KLD_loss = test(epoch, model, test_loader, reference_batch_3D_CFD, kl_weight, writer, device, args) # adding target sample to test method
            writer.add_scalar("test/test_loss", test_total_loss, epoch)
            writer.add_scalar("test/BCE_loss", test_BCE_loss, epoch)
            writer.add_scalar("test/KLD_loss", test_KLD_loss, epoch)
            print("Epoch [%d/%d] test_total_loss: %.3f, test_REC_loss: %.3f, test_KLD_loss: %.3f" % (epoch, args.epochs, test_total_loss, test_BCE_loss, test_KLD_loss))

            is_best = test_total_loss < best_test_loss
            best_test_loss = min(test_total_loss, best_test_loss)
            save_checkpoint({
                'epoch': epoch,
                'best_test_loss': best_test_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, is_best, outdir="results")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))

    # generate synthetic 3D cubes (TODO: generate scripts for generation)
    # print("[INFO] generating grid of synthetic 3D cubes from trained model")
    # plot_generation_grid(model, device, 9)
    writer.close()

    print()
    print("*************")
    print("FINISHED MAIN")
    print("*************")


if __name__ == '__main__':
    main()
