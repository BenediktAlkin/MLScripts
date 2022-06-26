import kappaprofiler as kp
import torch
import util

def main():
    model = util.Model()
    model.to(torch.device("cuda:0"))
    optimizer = util.get_optim(model)
    with kp.Stopwatch() as sw:
        util.train(model, optimizer)
    print(sw.elapsed_seconds)


if __name__ == "__main__":
    main()