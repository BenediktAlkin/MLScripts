import torch

def get_relative_position_index2d(win_h, win_w):
    # e.g. coords[:,0,0]=(0,0) coords[:,0,1]=(0,1) coords[:,4,3]=(4,3)
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    # flatten to (2, win_h*win_w)
    # e.g. coords[:,0]=(0,0) coords[:,1]=(0,1) coords[:,win_w]=(1,0)
    coords_flatten = torch.flatten(coords, 1)
    # calculate relative coordinates (2, win_h*win_w, win_h*win_w)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    # reshape to (win_h*win_w, win_h*win_w, 2)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    # shift to start from 0
    relative_coords[:, :, :] += win_h - 1
    # squash the last dimension from 2d to 1d without losing uniqueness of the index (win_h*win_w, win_h*win_w)
    relative_coords[:, :, 0] *= 2 * win_w - 1
    squashed_coords = relative_coords.sum(-1)
    return squashed_coords

def main():
    get_relative_position_index2d(7, 5)


if __name__ == "__main__":
    main()