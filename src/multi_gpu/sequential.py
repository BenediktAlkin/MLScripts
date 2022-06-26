import kappaprofiler as kp
import torch
import util

def main():
    model = util.Model()
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = util.get_optim(model)
    util.train(model, optimizer, only_single_batch=True)
    with kp.Stopwatch() as sw:
        util.train(model, optimizer, device=device)
    print(sw.elapsed_seconds)


if __name__ == "__main__":
    main()