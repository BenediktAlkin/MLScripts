import torch


def main():
    torch.manual_seed(5)
    model = torch.nn.Linear(10, 1)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    x = torch.randn(5, 10)
    x[0, 0] = float("inf")
    with torch.cuda.amp.autocast():
        y_hat = model(x)
        loss = loss_fn(y_hat, torch.randn_like(y_hat))
    print(f"loss: {loss}")
    scaled_loss = scaler.scale(loss)
    print(f"scaled_loss: {scaled_loss}")

    print(model.weight)
    optim.zero_grad()
    scaled_loss.backward()
    result = scaler.step(optim)
    print(f"scaler.step: {result}")
    #optim.step()
    scaler.update()
    print(model.weight)
    print(model.weight.grad)


if __name__ == "__main__":
    main()
