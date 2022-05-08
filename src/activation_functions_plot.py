import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

x = torch.linspace(-10, 10, 101)

plt.plot(x, F.relu(x), label="relu")
plt.plot(x, F.gelu(x), label="gelu")
plt.plot(x, F.silu(x), label="swish")

plt.legend()
plt.show()
