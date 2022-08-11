import torch
from torch import nn

mps_device = torch.device("mps")

def train_loop(verbose, device, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_loss = []
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        if device == 'GPU':
            X = X.to(mps_device)
            y = y.to(mps_device)
        # logits = model(X)
        # softmax = nn.Softmax(dim=1)
        # pred = softmax(logits)
        pred = model(X)
        loss = loss_fn(pred,y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch%100 == 0:
            loss, current = loss.item(), batch*len(X)
            correct +=(pred.argmax(1) == y).type(torch.float).sum().item()
            correct /= len(X)
            if verbose:
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            train_loss.append(loss)
            
    return correct, train_loss
    
def test_loop(verbose, device, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0
    
    with torch.no_grad():
        for X, y in dataloader:
            if device == 'GPU':
                X = X.to(mps_device)
                y = y.to(mps_device)
            # logits = model(X)
            # softmax = nn.Softmax(dim=1)
            # pred = softmax(logits)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct +=(pred.argmax(1) == y).type(torch.float).sum().item()
        
    test_loss /= num_batches
    correct /= size
    if verbose:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    
    return correct, test_loss