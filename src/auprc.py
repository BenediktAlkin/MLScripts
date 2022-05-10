# https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0118432&type=printable page 9
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
import torchmetrics

distributions = [
    ["random", dist.normal.Normal(0, 1), dist.normal.Normal(0, 1)],
    ["ER- (positives tend to have high scores)", dist.beta.Beta(4, 1), dist.beta.Beta(1, 1)],
    ["ER+ (negatives tend to have high scores)", dist.beta.Beta(1, 1), dist.beta.Beta(1, 4)],
    ["excellent", dist.normal.Normal(3, 1), dist.normal.Normal(0, 1)],
]
datasets = [
    [1000, 1000],
    [1000, 100],
    [1000, 10],
]
ds_scale = 1

for dist_name, dist_pos, dist_neg in distributions:
    for n_neg, n_pos in datasets:
        ds_name = f"{n_neg}-{n_pos} ({int(n_pos / n_neg * 100)})%"
        n_neg *= ds_scale
        n_pos *= ds_scale
        neg_scores = torch.randn(n_neg)  # dist_neg.sample(sample_shape=(n_neg,))
        pos_scores = torch.randn(n_pos)  # dist_pos.sample(sample_shape=(n_pos,))
        scores = torch.concat([neg_scores, pos_scores])
        targets = torch.concat([torch.zeros_like(neg_scores), torch.ones_like(pos_scores)]).type(torch.long)
        b_scores_rand = dist.normal.Normal(0, 1).sample(sample_shape=(n_neg + n_pos,))  # torch.randn_like(scores)
        b_scores_const = torch.zeros_like(scores)

        # metrics
        precision, recall, _ = torchmetrics.functional.precision_recall_curve(preds=scores, target=targets)
        auprc = torchmetrics.functional.auc(recall, precision)
        auroc = torchmetrics.functional.auroc(scores, targets)
        ap = torchmetrics.functional.average_precision(preds=scores, target=targets)
        print("calculated metrics")

        # numpy auprc
        # sanity check that AUPRC is calculated correctly
        # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
        # skr_precision, skr_recall, _ = sklearn.metrics.precision_recall_curve(targets, b_scores_rand)
        # skr_auprc = sklearn.metrics.auc(skr_recall, skr_precision)
        # print(f"sklearn auprc {skr_auprc}")

        # baselines (random)
        br_precision, br_recall, _ = torchmetrics.functional.precision_recall_curve(preds=b_scores_rand, target=targets)
        br_auprc = torchmetrics.functional.auc(br_recall, br_precision)
        br_auroc = torchmetrics.functional.auroc(b_scores_rand, targets)
        br_ap = torchmetrics.functional.average_precision(preds=b_scores_rand, target=targets)
        print("calculated random baseline")
        # baselines (constant)
        res = torchmetrics.functional.precision_recall_curve(preds=b_scores_const, target=targets)
        bc_precision, bc_recall, _ = res
        bc_auprc = torchmetrics.functional.auc(bc_recall, bc_precision)
        bc_auroc = torchmetrics.functional.auroc(b_scores_const, targets)
        bc_ap = torchmetrics.functional.average_precision(preds=b_scores_const, target=targets)
        print("calculated const baseline")

        # plots
        # clip scores at 0.05 and 0.9 quantile
        q_lower = torch.quantile(scores, 0.05)
        q_upper = torch.quantile(scores, 0.95)
        rescaled_scores = scores.clip(q_lower, q_upper)
        # scale scores to [-1, 1]
        rescaled_scores = rescaled_scores - rescaled_scores.min()
        rescaled_scores = rescaled_scores / rescaled_scores.max()
        rescaled_scores = rescaled_scores * 2 - 1
        # y coordinate is uniformly sampled for better visualization
        y = torch.rand_like(rescaled_scores)
        # scatter
        inlier_idxs = targets == 0
        outlier_idxs = targets == 1
        plt.clf()
        plt.scatter(rescaled_scores[inlier_idxs], y[inlier_idxs], marker='^', label="inlier")
        plt.scatter(rescaled_scores[outlier_idxs], y[outlier_idxs], marker='o', label="outlier")
        plt.title("\n".join([
            f"scoring function={dist_name}",
            f"inliers={n_neg} outliers={n_pos} ({int(n_pos / n_neg * 100)}%)"
        ]))
        for text in [
            f"AUROC={auroc:.2f}",
            f"AUPRC={auprc:.2f} random={br_auprc:.2f} const={bc_auprc:.2f}",  # skr={skr_auprc:.2f}",
            f"AP={ap:.2f} random={br_ap:.2f} const={bc_ap:.2f}",
        ]:
            plt.plot([], [], ' ', label=text)
        plt.ylabel("uniform sampled y dimension")
        plt.xlabel("outlier score")
        plt.legend(loc="upper left")
        plt.savefig(f"dist={dist_name} ds={ds_name}.png")

        print("-------------------------------------")
        print(f"dist={dist_name} ds={ds_name}")
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPRC: {auprc:.4f}  random: {br_auprc:.4f}  const: {bc_auprc:.4f}")
        print(f"   AP: {ap:.4f}  random: {br_ap:.4f}  const: {bc_ap:.4f}")
