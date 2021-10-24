import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

# Hyperparameters
learning_rate = 1e-3

# Chen CNN
class CNN(nn.Module):
    def _init_(self, num_classes: int = 4):
        super(CNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 1, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(64),
            nn.Conv2d(64, 192, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding =1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding =1),
            nn.ReLU(),
            nn.MaxPool2d(256),
            nn.AdaptiveAvgPool2d((6,6)),
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.main(x)

# Initialize network
model = CNN()
print(model)

# Get our Dataframe

# Loss and optimizer
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# optional SGD optimizer
# optimizer = optim.SGD(model.parameters(), lr=1e-3)

"""
def train(df, model, loss_fn, optimizer):
    # may be different
    size = len(df)
    model.train()
    # modify this based on dataframe data
    for batch, (X, y) in df:
        pred = model.forward(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item, batch * len(X)
            # output the current loss
            print(f"loss: {loss:>7f}, [{current:>5d}/{size:>5d}]")

def test(df, model, loss_fn, num_batches):
    size = len(df)
    model.eval()
    test_loss, correct = 0, 0
    # pain
    with torch.no_grad():
        for X, y in df:
            pred = model.forward(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

# model training
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
"""
