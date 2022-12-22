import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import time
from common.focal_loss import FocalLoss
from sklearn.metrics import f1_score


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
    torch.save(net.state_dict(), 'checkpoints/epoch_' + str(epoch) + '.pth')


def plot_graph(train_all, val_all, title):
    epochs = range(1, len(train_all) + 1)
    plt.plot(epochs, train_all, 'bo', label='training ' + title)
    plt.plot(epochs, val_all, 'b', label='validation ' + title)
    plt.title('Training and validation ' + title)
    plt.xlabel('epochs') 
    plt.ylabel(title) 
    plt.legend()
    plt.savefig('plots/' + title + '.png')
    plt.clf()


def acc(preds, targs, threshold):
    preds = (preds > threshold).int()
    targs = targs.int()
    return (preds==targs).float().mean()


def batch_f1_score(preds, targs, threshold):
    preds = (preds > threshold).int().cpu()
    targs = targs.int().cpu()
    f1_scores = [f1_score(targs[i], preds[i], average='macro') for i in range(len(preds))]
    return np.array(f1_scores).mean()


def evaluate_accuracy(net, data_iter, loss, device, threshold):
    net.eval()  # Set the model to evaluation mode

    total_loss = 0
    total_hits = 0
    total_f1 = 0
    total_samples = 0
    
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            total_loss += float(l)
            total_hits += acc(y_hat, y, threshold)
            total_samples += 1
            total_f1 += batch_f1_score(y_hat, y, threshold)

    val_loss = float(total_loss) / len(data_iter)
    val_acc = float(total_hits) / total_samples
    val_f1 = float(total_f1) / total_samples
    return val_loss, val_acc, val_f1


def train_epoch(net, train_iter, loss, optimizer, device, threshold):  
    # Set the model to training mode
    net.train()
    # Sum of training loss, sum of training correct predictions, no. of examples
    total_loss = 0
    total_hits = 0
    total_f1 = 0
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
        total_hits += acc(y_hat, y, threshold)
        total_f1 += batch_f1_score(y_hat, y, threshold)
        total_samples += 1
    
    train_loss = float(total_loss) / len(train_iter)
    train_acc = float(total_hits) / total_samples
    train_f1 = float(total_f1) / total_samples
    return train_loss, train_acc, train_f1


def train(net, train_iter, val_iter, num_epochs, lr, threshold, device, save_model):
    """Train a model."""
    train_loss_all = []
    train_acc_all = []
    train_f1_all = []
    val_loss_all = []
    val_acc_all = []
    val_f1_all = []
    print('Training on', device)
    net.to(device)
    print("Freezing the resnet layers for the first 10 epochs...")
    for param in net.encoder.parameters():
        param.requires_grad = False
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = FocalLoss()
    min_val_loss = 10000.0
    max_f1_score = 0
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, train_acc, train_f1 = train_epoch(net, train_iter, loss, optimizer, device, threshold)
        end = time.time()
        print(f"Training time: {end - start}s")
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        train_f1_all.append(train_f1)
        start = time.time()
        val_loss, val_acc, val_f1 = evaluate_accuracy(net, val_iter, loss, device, threshold)
        end = time.time()
        print(f"Evaluate on validation time: {end - start}s")
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)
        val_f1_all.append(val_f1)
        print(f'Epoch {epoch + 1}, \nTrain loss {train_loss:.4f}, Train accuracy {train_acc:.4f}, Train F1 score {train_f1:.4f}\nValidation loss {val_loss:.4f}, Validation accuracy {val_acc:.4f}, Validation F1 score {val_f1:.4f}')

        # save model if it has lower loss and save_model equals True
        if (val_loss < min_val_loss or val_f1 > max_f1_score) and save_model:
            min_val_loss = val_loss
            max_f1_score = val_f1
            save_model_weights(net, epoch)
        
        plot_graph(train_loss_all, val_loss_all, 'loss')
        plot_graph(train_acc_all, val_acc_all, 'accuracy')
        plot_graph(train_f1_all, val_f1_all, 'f1_score')

        # Unfreeze resnet layers after 10 epochs
        if epoch == 10:
            print("Unfreezing the resnet layers")
            for param in net.encoder.parameters():
                param.requires_grad = True
