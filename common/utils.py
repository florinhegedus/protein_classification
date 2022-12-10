import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from common.focal_loss import FocalLoss


def display_image(img_path):
    colors = ['red', 'green', 'blue']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(img_path + '_' + color + '.png', flags).astype(
                        np.float32) for color in colors]
    img = np.stack(img, axis=-1)
    plt.imshow(img, interpolation='nearest')
    plt.show()

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def save_model_weights(net, epoch):
    print("Saving model...")
    torch.save(net.state_dict(), str(epoch) + '.pth')

def acc(preds, targs, threshold=0.0):
    preds = (preds > threshold).int()
    targs = targs.int()
    return (preds==targs).float().mean()

def evaluate_accuracy(net, data_iter, loss, device):
    """Compute the accuracy for a model on a dataset."""
    net.eval()  # Set the model to evaluation mode

    total_loss = 0
    total_hits = 0
    total_samples = 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            total_loss += float(l)
            total_hits += acc(y_hat, y)
            total_samples += y.numel()
    return float(total_loss) / len(data_iter), float(total_hits) / total_samples  * 100

def train_epoch(net, train_iter, loss, optimizer, device):  
    # Set the model to training mode
    net.train()
    # Sum of training loss, sum of training correct predictions, no. of examples
    total_loss = 0
    total_hits = 0
    total_samples = 0
    steps = 0
    for X, y in train_iter:
        # Compute gradients and update parameters
        steps += 1
        if steps % 10 == 0:
            print("   step " + str(steps) + "/" + str(len(train_iter)), end="\r")
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        # Using PyTorch built-in optimizer & loss criterion
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        total_loss += float(l)
        total_hits += acc(y_hat, y)
        total_samples += y.numel()
    # Return training loss and training accuracy
    return float(total_loss) / len(train_iter), float(total_hits) / total_samples  * 100

def train(net, train_iter, val_iter, num_epochs, lr, device, save_model):
    """Train a model."""
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    print('Training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = FocalLoss()
    min_val_loss = 10000.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, optimizer, device)
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss, val_acc = evaluate_accuracy(net, val_iter, loss, device)
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)
        print(f'Epoch {epoch + 1}, Train loss {train_loss:.2f}, Train accuracy {train_acc:.2f}, Validation loss {val_loss:.2f}, Validation accuracy {val_acc:.2f}'

        # save model if it has lower loss and save_model equals True
        if val_loss < min_val_loss and save_model:
            min_val_loss = val_loss
            save_model_weights(net, epoch)

    return train_loss_all, train_acc_all, val_loss_all, val_acc_all