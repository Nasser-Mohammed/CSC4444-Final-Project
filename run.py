import torch
import matplotlib.pyplot as plt
from model2 import create_model
from PIL import Image
import torchvision.transforms as transforms
import os
import pickle
import argparse
from dataloader import create_dataloader


def testData(model):
    _, test_data, y = create_dataloader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===================================================")
    print(f"Testing model on test set")
    model.eval()
    model = model.to(device)
    test_correct = 0
    ttl = 0
    with torch.no_grad():
        for input, label in test_data:
            input, label = input.to(device), label.to(device)
            val_pred = model(input)
            _, predicted = torch.max(val_pred, 1)
            test_correct += (predicted == label).sum().item()
            ttl += label.size(0)
            acc = (test_correct/ttl)*100
        print(f"Test Accuracy: {acc:.4f}%")
    print("===================================================")



def testDog(imgPath, model, label = None):
    img = Image.open(imgPath) 
    print("you inputted the following image: ")
    plt.imshow(img)
    plt.show()
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by your model
    transforms.ToTensor(),          # Convert to a PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalization (example for ImageNet)
    ])
    num2LabelDict = {}
    # Apply the transformation to the image
    img = transform(img) 
    
    img = torch.stack([img], dim=0)
    with torch.no_grad():
        pred = model(img)
    if os.path.isfile("num2LabelDict.pkl"):
        with open('num2LabelDict.pkl', 'rb') as f:  # 'rb' for binary read mode
            num2LabelDict = pickle.load(f)
    _, predicted_class = torch.topk(pred, 3, dim = 1, sorted = True)
    predicted_class = predicted_class[0]
    output = num2LabelDict[predicted_class[0].item()]
    #print(f"model predicted: {output}: correct answer was: {label}")
    print("=========================================================================")
    print(f"Model predicted the following breeds in order of likeliness for: {label}")
    for i, pred in enumerate(predicted_class):
        print(f"{i+1}: {num2LabelDict[pred.item()]}")
        #print(f"{num2LabelDict[i[1].item()]} was {pred[0, i[1]].item()}")

    print("=========================================================================")
    return output



def main():
    print("Enter one of the following numbers:")
    print("[1] Test model on test data")
    print("[2] Test model on a few sample dog images")
    print("[3] Test model on your own image")
    choice = input()
    if choice.isnumeric():
        choice = int(choice)
    else:
        exit("invalid input")
    model = create_model()
    model.load_state_dict(torch.load('model_state_dict.pth'))
    model.eval()

    if choice == 1:
        testData(model)
    elif choice == 2:
        testDog("sb.jpg", model, "Saint Bernard")
        testDog("dal.jpg", model, "Dalmation")
        testDog("bt.jpg", model, "Boston Terrier")
        testDog("corgi.jpg", model, "Corgi")
    elif choice == 3:
        print("Please enter the path to your own image below, make sure the file is in the current folder and that there are no spaces in the name, do not use quotes")
        path = input()
        if os.path.isfile(path):
            print("now enter the name of the breed of the input image")
            label = input()
            testDog(path, model, label)
        else:
            print("cannot locate your file, make sure it is in the same folder as this file")
    else:
        print('you did not enter one of the above options')
        exit("incorrect input")


if __name__ == '__main__':
    main()