import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from loss import loss_function
from utils import plot_reconstruction_grid
# from utils import (save_reconstructions, save_representations, save_projected_representations, plot_reconstruction_grid)

def test(epoch, model, test_loader, reference_batch, kl_weight, writer, device, args):
    """
    Evaluates reconstructions at every epoch (at batch idx 0) by loading test data
    and feeding it through the 3D CVAE.

    TODO: Evaluate generations at every epoch.
    """

    model.eval()
    test_total_loss = 0
    test_BCE_loss = 0
    test_KLD_loss = 0

    # print()
    print("[INFO] entered batch testing")
    print("test device:", device)
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader), total=len(test_loader), desc='test'):
            # forward pass for random batch
            data = data.to(device, dtype=torch.float)
            recon_batch, mu, logvar, latent_batch = model(data)
            total_loss, BCE_loss, KLD_loss = loss_function(recon_batch, data, mu, logvar, kl_weight)

            test_total_loss += total_loss.item()
            test_BCE_loss += BCE_loss.item()
            test_KLD_loss += KLD_loss.item()

            # forward pass for reference batch
            reference_batch = reference_batch.to(device, dtype=torch.float)
            reference_recon_batch, _, _, _ = model(reference_batch)

            if batch_idx == 0:
                # working only with reference batch for now
                print("calling plot_reconstruction_grid() to save reconstructions")
                # plot_reconstruction_grid(data, recon_batch, 3, 21, epoch, save_grid=True)
                plot_reconstruction_grid(reference_batch, reference_recon_batch, 3, 21, epoch, save_grid=True, reference_batch=True)

                # save_reconstructions(data, recon_batch, epoch, writer, args)
                # save_representations(latent_batch, epoch, writer, args)
                # save_projected_representations(latent_batch, epoch, writer, args, download=True)

    test_total_loss /= len(test_loader.dataset)
    test_BCE_loss /= len(test_loader.dataset)
    test_KLD_loss /= len(test_loader.dataset)

    return test_total_loss, test_BCE_loss, test_KLD_loss
