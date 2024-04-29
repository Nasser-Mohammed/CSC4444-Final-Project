

from dataloader import create_dataloader#import create_dataloader
from model2 import create_model 
import torch.nn as nn
import torch.optim as optim
import torch
import time
import matplotlib.pyplot as plt
import pickle
import os

def train_model(model, optimizer, loss_fn, train_data, validation_data, device, epochs = 5):
    label2NumDict, num2LabelDict = train_data.dataset._getLabelDict()
    # Save the dictionary to a file
    if not os.path.isfile("label2NumDict.pkl"):
        print('writing dict 1')
        with open('label2NumDict.pkl', 'wb') as f:  # 'wb' for binary write mode
            pickle.dump(label2NumDict, f)
    if not os.path.isfile("num2LabelDict.pkl"):
        print('writing dict 2')
        with open('num2LabelDict.pkl', 'wb') as f:  # 'wb' for binary write mode
            pickle.dump(num2LabelDict, f)
    fitCnt = 0
    valAccuracies = []
    bestTrainAcc = 0
    bestEpoch = 0
    bestValAcc = 0
    bestLoss = float('inf')
    savedTrainAcc = 0
    savedValAcc = 0
    savedEpoch = 0
    savedLoss = float('inf')
    print("Beginning training")
    print("===================================================")
    for epoch in range(epochs):
        model.train()
        print(f"EPOCH: {epoch}/{epochs}")
        start = time.time()
        runningLoss = 0
        train_correct = 0 #num of correct predictions
        total = 0 #num of predictions
        i = 0
        for input, label in train_data:
            i += 1
            if i %25 == 0:
                print(f"step {i}/{len(train_data)}")
            #input is: 32x3x224x224
            #label is: [32]
            #0 index is which image from batch
            #1 index is which dimension of RGB to pick
            #2 and 3 index are the actual pixels of the image
            input, label = input.to(device), label.to(device)
            #plt.imshow(input[0].permute(1, 2, 0).cpu())
            #plt.show()
            #print(input[0])
            #tmp = label[0].cpu().item()
            #tmp2 = label[1].cpu().item()
            #crt1 = num2LabelDict[tmp]
            #crt2 = num2LabelDict[tmp2]
            #print(f"label 1 is: {num2LabelDict[tmp]}")
            #print(f"label 2 is: {num2LabelDict[tmp2]}")
            optimizer.zero_grad()
            y_pred = model(input)
            loss = loss_fn(y_pred, label)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(y_pred, 1)
            #print(f"for {crt1} model guessed: {num2LabelDict[predicted[0].item()]}| for {crt2} model guessed: {num2LabelDict[predicted[1].item()]}")
            train_correct += (predicted == label).sum().item()
            total += label.size(0)
            runningLoss += loss.item()
        end = time.time()
        print(f"Epoch {epoch+1}/{epochs} summary")
        print("===================================================")
        print(f"Accuracy: {train_correct / total:.4f}")
        print(f"Loss: {runningLoss/len(train_data):.4f}")
        print(f"Time: {end - start}")
        print("===================================================")

        if epoch%5 == 0:
            print("===================================================")
            print(f"Testing model on validation set")
            model.eval()
            val_correct = 0
            ttl = 0
            with torch.no_grad():
                for input, label in validation_data:
                    input, label = input.to(device), label.to(device)
                    val_pred = model(input)
                    _, predicted = torch.max(val_pred, 1)
                    val_correct += (predicted == label).sum().item()
                    ttl += label.size(0)
                    valAccuracies.append(val_correct/ttl)
                print(f"Validation Accuracy: {val_correct/ttl:.4f}")
            print("===================================================")

            if val_correct/ttl > bestValAcc:
                bestTrainAcc = train_correct/total
                bestEpoch = epoch
                bestLoss = runningLoss/len(train_data)
                bestValAcc = val_correct/ttl

        if epoch >= 9:
            if valAccuracies[-1] < valAccuracies[-2] and epoch%5 == 0:
                fitCnt += 1 

            else:
                fitCnt = 0
                #checks if this is best round for val accuracy
            if bestValAcc == val_correct/ttl and epoch%5 == 0:
                torch.save(model.state_dict(), 'model_state_dict.pth')
                savedEpoch = epoch
                savedLoss = runningLoss/len(train_data)
                savedTrainAcc = train_correct/total
                savedValAcc = val_correct/ttl

        if fitCnt > 2:
            print("Model is overfitting: stopping training")
            print(f"final stats: saved[train acc: {savedTrainAcc}, train loss: {savedLoss}, val acc: {savedValAcc}, epoch: {savedEpoch}]")
            print(f"best stats: train acc: {bestTrainAcc}, train loss: {bestLoss}, val acc: {bestValAcc}, epoch: {bestEpoch}")
            return
    print(f"final stats: saved[train acc: {savedTrainAcc}, train loss: {savedLoss}, val acc: {savedValAcc}, epoch: {savedEpoch}]")
    print(f"best stats: train acc: {bestTrainAcc}, train loss: {bestLoss}, val acc: {bestValAcc}, epoch: {bestEpoch}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"You are training on: {device}")
    train_dataloader, test_dataloader, validation_dataloader = create_dataloader()
    model = create_model().to(device)
    model.load_state_dict(torch.load('bestModel.pth'))
    lr = 0.0001
    momentum = 0.9
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)# optim.Adam(model.parameters(), lr = lr)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 50
    train_model(model, optimizer, loss_fn, train_dataloader, validation_dataloader, device, epochs = epochs)


    print('success')

if __name__ == '__main__':
    main()