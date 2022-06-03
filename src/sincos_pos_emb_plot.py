import torch
import matplotlib.pyplot as plt

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float)
    grid_w = torch.arange(grid_size, dtype=torch.float)
    grid = torch.meshgrid(grid_w, grid_h)  # here w goes first
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.concat([np.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.concat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = out.sin()  # (M, D/2)
    emb_cos = out.cos()  # (M, D/2)

    emb = torch.concat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def plot(emb, colorbar=False):
    plt.imshow(emb)
    plt.ylabel("position")
    plt.xlabel("dimension")
    if colorbar:
        plt.colorbar()
    plt.show()


def main():
    positions = 128
    emb_dim = 192
    pos = torch.arange(positions, dtype=torch.float32).reshape(1, 1, positions)
    emb = get_1d_sincos_pos_embed_from_grid(emb_dim, pos)
    plot(emb)

    # for i in range(positions):
    #     half_emb_dim = int(emb_dim / 2)
    #     plt.plot(range(half_emb_dim), emb[i, :half_emb_dim] + (i * 2))
    #     plt.plot(range(half_emb_dim, emb_dim), emb[i, half_emb_dim:] + (i * 2))
    # plt.ylabel("positions")
    # plt.xlabel("dimensions")
    # plt.show()
    # plt.clf()

    # plot learned embedding
    state_dict = torch.load("res/VitConcatProjPosLearnable cp=E80U6960S3563520.th", map_location=torch.device("cpu"))
    pos_emb = state_dict["pos_embed"]
    pos_emb = pos_emb[0]  # remove unnecessary dimension for broadcasting
    pos_emb = pos_emb[1:, :]  # remove cls token embedding (only 0s)
    pos_emb_h, pos_emb_w = pos_emb.chunk(2, dim=1)  # split into heigt and width embedding
    plot(pos_emb_h)
    plot(pos_emb_w)

    # plot reference embedding
    positions, dimensions = pos_emb.shape
    ref_emb = get_2d_sincos_pos_embed(dimensions, int(positions ** 0.5))
    pos_emb_h_ref, pos_emb_w_ref = ref_emb.chunk(2, dim=1)
    plot(pos_emb_w_ref)
    plot(pos_emb_h_ref)


    print("fin")


if __name__ == "__main__":
    main()