import torch
import torch.nn as nn

import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

def train(model, train_loader, epoch, optimizer, criterion):
    model.train()
    print('\nEpoch: %d' % epoch)
    running_loss = 0
    correct = 0
    total = 0
    for batch_idx, (image, label) in enumerate(train_loader):
        # image, label = image.to(device), label.to(device)
        image = image.view(-1, 28 * 28)
        
        optimizer.zero_grad()
        outputs = model(image)

        loss = criterion(outputs, label)
        # running_loss += loss.item()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(label).sum().item()
        total += label.size(0)

    train_loss = running_loss / len(train_loader)
    
    return train_loss
    # print('Loss:{} | Acc:{} ({}/{})'.format((train_loss/(batch_idx+1)), 100.*correct/total, correct, total))


def valid(test_loader, model, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.view(-1, 28 * 28)

            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / len(test_loader)
    val_acc = float(correct) / total
    
    return val_loss, val_acc
 
def test(model, test_loader, epoch, criterion):
    model.eval()
    print('\nEpoch: %d' % epoch)
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx,(image, label) in enumerate(test_loader):
        # image, label = image.to(device), label.to(device)
        image = image.view(-1, 28 * 28)

        outputs = model(image)
        loss = criterion(outputs, label)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()
    
    test_loss = test_loss / len(test_loader)
    test_acc = float(correct) / total
    
    return test_loss, test_acc        
    # print('Loss:{} | Acc:{} ({}/{})'.format((test_loss/(batch_idx+1)), 100.*correct/total, correct, total))
