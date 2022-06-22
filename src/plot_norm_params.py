import matplotlib.pyplot as plt
import torch

path = "C:/Users/Benedikt Alkin/Documents/pretrained/MAE_ImageNet/224x224_encoder_pretrain_vit_base.pth"

def main():
    state_dict = torch.load(path)
    weights = {}
    biases = {}
    for key in state_dict.keys():
        if "norm" in key:
            if ".weight" in key:
                weights[key.replace(".weight", "")] = state_dict[key].mean()
            if ".bias" in key:
                biases[key.replace(".bias", "")] = state_dict[key].mean()

    plt.plot(range(len(weights)), weights.values(), label="weight")
    plt.plot(range(len(biases)), biases.values(), label="bias")
    plt.xticks(range(len(weights)), weights.keys(), rotation=45, ha="right", rotation_mode="anchor")
    plt.ylim(-1, 1)
    plt.xlim(-1, len(weights))
    plt.hlines(0, xmin=-1, xmax=len(weights), colors="black", alpha=0.6)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    # plt.savefig("plot_norm_params.png")


if __name__ == "__main__":
    main()