import kappaprofiler as kp
import torch
import torch.nn as nn
import yaml
import util

def main():
    model = util.Model()
    model = nn.DataParallel(model)
    model.to(torch.device("cuda:0"))
    optimizer = util.get_optim(model)
    util.train(model, optimizer, only_single_batch=True)
    with kp.Stopwatch() as sw:
        util.train(model, optimizer)
    print(sw.elapsed_seconds)


if __name__ == "__main__":
    main()