import torch

from torchvision.models.resnet import  BasicBlock
from torchvision.models import ResNet
from torch.utils.data import DataLoader
from Dataset import PorousData
import torch.nn as nn
import torch.optim as optim

from utils import data_transform, show_examples
from Args import Args

# Args
args = Args()
USE_GPU = Args.USE_GPU
n_epochs = Args.n_epochs
batch_size_train = Args.batch_size_train
batch_size_test = Args.batch_size_test
learning_rate = Args.learning_rate
log_interval = Args.log_interval
random_seed = Args.random_seed

torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)

# data loader
data_train = PorousData(Args.train_data_path, transform=data_transform['train'])
data_test = PorousData(Args.test_data_path, transform=data_transform['test'])
train_loader = DataLoader(data_train, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(data_test, batch_size=batch_size_test, shuffle=False)

# show examples
if Args.show_examples:
    show_examples(test_loader, data_train)

# Model and optimizer
network = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=1)
optimizer = optim.Adam(network.parameters(), lr=learning_rate,
                       betas=[0.9, 0.99])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # learning rate decay
criterion = nn.BCEWithLogitsLoss()

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

if USE_GPU:
    network = network.cuda()
    criterion = criterion.cuda()


# train
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if USE_GPU:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target.view(-1, 1).float())
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), args.model_save_path[0])
            torch.save(optimizer.state_dict(), args.model_save_path[1])
    scheduler.step()


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            if USE_GPU:
                data = data.cuda()
                target = target.cuda()

            output = network(data)
            test_loss += criterion(output, target.view(-1, 1).float()).item()
            output_logit = torch.sigmoid(output.data)
            pred = output_logit > 0.6 # threshold for prediction

            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
