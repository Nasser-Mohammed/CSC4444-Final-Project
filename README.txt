README
The project folder is: 4444 Project
Inside you will find run.py, this is the script to run if you want to interact with the program. It will present you different options that explain what to do.
Make sure you cd into the 4444 Project folder before attempting to run.

==================================================================================
TO RUN: python run.py (or python3 run.py depending on your system) in your terminal
==================================================================================

It will offer you several prompts, they should be explanatory and clear on what to do. You will be able to test the neural net on our test set, some sample images from the internet, or your own image of a dog that you'd like to test.
When inputting the path to your own image, do not use quotes, and make sure the file is in the same directory as the run.py file. If the name of the file has spaces, rename it. 

train.py is the script we used to train the neural net, I would not recommend running this, however, it should load the model and start training if you do.
dataloader.py is the file we used to make our custom data set class and dataloaders for iterating the data
model2.py is our revised neural net file
model.py is our old neural net file
anything ending in .pth are different saved versions of the model
anything ending in .pkl is data that is needed to run the run.py file

Dataset:
Our dataset can be found in the subdirectory called data, so /4444 Project/data/archive(1), there it is split into training, testing, and validation sets, with the index file also being found there.

Environment:
in the 4444 Project folder, you will also find my environment configuration, called: environment.yaml
to replicate this environment enter the following command if you have conda: conda env create -f environment.yaml
make sure you are in the 4444 Project folder before running this