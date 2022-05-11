import matplotlib.pyplot as plt
import torch
import torchmetrics.functional

outlier_ratio = 0.01
dataset_size = 10000
targets = torch.concat([
    torch.zeros([int(dataset_size * (1 - outlier_ratio))]),
    torch.ones([int(dataset_size * outlier_ratio)]),
]).type(torch.long)

# scores = torch.randn(*targets.shape)
scores = torch.zeros(targets.shape)
precision, recall, pr_thresholds = torchmetrics.functional.precision_recall_curve(preds=scores, target=targets)
auprc = torchmetrics.functional.auc(recall, precision)

# plot
plt.plot(recall, precision, label=f"AUPRC={auprc:.4f}")
plt.legend()
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
# plt.show()
plt.savefig("test1.png")
