import os
import torch
import shutil

def save_checkpoint(state, is_best, outdir="results"):
    """
    Saves models parameters into outdir.
    Note: for 3D models the .pth file can be 100MB size.
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    checkpoint_file = os.path.join(outdir, "checkpoint.pth")
    best_file = os.path.join(outdir, "model_best.pth")

    torch.save(state, checkpoint_file)

    if is_best:
        shutil.copyfile(checkpoint_file, best_file)
