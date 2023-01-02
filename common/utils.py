import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
import time
import json
from sklearn.metrics import f1_score
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


def update_metrics(metrics, train_metrics, val_metrics):
    metrics['train_loss'].append(train_metrics[0])
    metrics['train_acc'].append(train_metrics[1])
    metrics['train_f1'].append(train_metrics[2])
    metrics['val_loss'].append(val_metrics[0])
    metrics['val_acc'].append(val_metrics[1])
    metrics['val_f1'].append(val_metrics[2])

    with open("metrics.json", "w") as file:
        json.dump(metrics, file)

    return metrics


def train(net, train_iter, val_iter, num_epochs, lr, threshold, device, save_model, continue_training):
    """Train a model."""
    if continue_training:
        with open("metrics.json") as file:
            metrics = json.load(file)
        continue_from_epoch = len(metrics['train_loss'])
    else:
        metrics = {'train_loss': [], 'train_acc': [], 'train_f1': [],
                    'val_loss': [], 'val_acc': [], 'val_f1': []}
        continue_from_epoch = 0
    
    print('Training on', device)
    net.to(device)
    if continue_from_epoch < 10:
        print("Freezing the resnet layers for the first 10 epochs...")
        for param in net.encoder.parameters():
            param.requires_grad = False
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = FocalLoss()
    min_val_loss = 10000.0
    max_f1_score = 0
    for epoch in range(continue_from_epoch, num_epochs, 1):
        print(f"---Epoch {epoch}---")
        start = time.time()
        train_loss, train_acc, train_f1 = train_epoch(net, train_iter, loss, optimizer, device, threshold)
        end = time.time()
        print(f"Training time: {end - start}s")
        start = time.time()
        val_loss, val_acc, val_f1 = evaluate_accuracy(net, val_iter, loss, device, threshold)
        end = time.time()
        print(f"Evaluate on validation time: {end - start}s")
        print(f'Train loss {train_loss:.4f}, Train accuracy {train_acc:.4f}, Train F1 score {train_f1:.4f}\nValidation loss {val_loss:.4f}, Validation accuracy {val_acc:.4f}, Validation F1 score {val_f1:.4f}')

        # update metrics
        train_metrics = (train_loss, train_acc, train_f1)
        val_metrics = (val_loss, val_acc, val_f1)
        metrics = update_metrics(metrics, train_metrics, val_metrics)

        # save model if it has lower loss and save_model equals True
        if (val_loss < min_val_loss or val_f1 > max_f1_score) and save_model:
            min_val_loss = val_loss
            max_f1_score = val_f1
            save_model_weights(net, epoch)
        
        plot_graph(metrics['train_loss'], metrics['val_loss'], 'loss')
        plot_graph(metrics['train_acc'], metrics['val_acc'], 'accuracy')
        plot_graph(metrics['train_f1'], metrics['val_f1'], 'f1_score')

        # Unfreeze resnet layers after 10 epochs
        if epoch == 10:
            print("Unfreezing the resnet layers")
            for param in net.encoder.parameters():
                param.requires_grad = True


def postprocess(y_hat, img_labels, idx, threshold):
    for sample in y_hat:
        res = (sample > threshold).int()
        s = ' '.join([str(i) for i in np.nonzero(res.cpu().numpy())[0]])
        img_labels.iloc[idx, 1] = s
        idx += 1

    return img_labels, idx


def test(net, test_iter, img_labels, threshold, device):
    """Test a model."""
    print('Testing on', device)
    net.to(device)
    net.eval()  # Set the model to evaluation mode
    idx = 0
    
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            img_labels, idx = postprocess(y_hat, img_labels, idx, threshold)
            print(f"    {idx/16}/{len(test_iter)}", end="\r")

    img_labels.to_csv('sample_submission.csv', index=False)