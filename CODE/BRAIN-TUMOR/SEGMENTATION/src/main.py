import torch.optim as optim
from model import UNet

def train(model, optimizer, loss_fn, df, epochs):
    for i in range(len(epochs)):
        for i, data in enumerate(df):
            x, y = data
            optimizer.zero_grad()

            outp = model(x)
            loss = loss_fn(outp, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(i, running_loss)

def test(model, df):
    # TODO write test method
    pass

def main():
    # I hope this is really not shit dude
    los = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

if __name__ == "__main__":
    main()
