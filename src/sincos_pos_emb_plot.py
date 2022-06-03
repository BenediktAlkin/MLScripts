import torch
import matplotlib.pyplot as plt

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

def main():
    positions = 128
    emb_dim = 192
    pos = torch.arange(positions, dtype=torch.float32).reshape(1, 1, positions)
    emb = get_1d_sincos_pos_embed_from_grid(emb_dim, pos)

    # for i in range(positions):
    #     half_emb_dim = int(emb_dim / 2)
    #     plt.plot(range(half_emb_dim), emb[i, :half_emb_dim] + (i * 2))
    #     plt.plot(range(half_emb_dim, emb_dim), emb[i, half_emb_dim:] + (i * 2))
    # plt.ylabel("positions")
    # plt.xlabel("dimensions")
    # plt.show()
    # plt.clf()

    plt.imshow(emb)
    plt.ylabel("position")
    plt.xlabel("dimension")
    plt.colorbar()
    plt.show()

    print("fin")


if __name__ == "__main__":
    main()