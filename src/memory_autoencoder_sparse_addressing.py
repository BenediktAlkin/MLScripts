import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def main():
    lambda_ = 15
    eps = 1e-6

    # equation 6 of https://arxiv.org/pdf/1904.02639.pdf
    w_hat_fn = lambda w: F.relu(w - lambda_) * w / ((w - lambda_).abs() + eps)

    x = torch.linspace(-100, 100, 1001)

    plt.plot(x, w_hat_fn(x))

    plt.show()

if __name__ == "__main__":
    main()