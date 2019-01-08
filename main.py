from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from c4 import C4
from confm import ConfusionMeter

##########################################################################
# Network

class Net(nn.Module):

    def __init__(self, d=1, nclass=3):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6*d, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(6*d),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(6*d, 16*d, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(16*d),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.classif = nn.Sequential(
            nn.Linear(16*2*2*d, 120*d, bias=False),
            nn.BatchNorm1d(120*d),
            nn.ReLU(inplace=True),
            nn.Linear(120*d, 84*d, bias=False),
            nn.BatchNorm1d(84*d),
            nn.ReLU(inplace=True),
            nn.Linear(84*d, nclass)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x)
        return F.log_softmax(x, dim=1)

##########################################################################
# Engine

def train(args, model, device, train_loader, optimizer, epoch):
    confm = ConfusionMeter(3)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        confm.add(output.detach(), target.detach())
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    confm.show()

def test(args, model, device, test_loader):
    confm = ConfusionMeter(3)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            confm.add(output, target)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    confm.show()

##########################################################################
# Main

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='C4')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--pc-train', type=float, default=.7, metavar='PC',
                        help='pourcentage of items in the training set (default: .7)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1337, metavar='S',
                        help='random seed (default: 1337)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_set = C4('data', split='train', pc_train=args.pc_train, seed=args.seed)
    test_set = C4('data', split='val', pc_train=args.pc_train, seed=args.seed)

    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"c4_cnn.pt")
        
if __name__ == '__main__':
    main()