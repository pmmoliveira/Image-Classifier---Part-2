import time
from PIL import Image
import numpy as np


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from workspace_utils import active_session, keep_awake
from get_input_args_train import get_input_args_train
import json

with open('ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
in_arg = get_input_args_train()

# Print command line arguments
print("Command line arguments:\n dir = " + in_arg.dir + "\n save_dir = " + in_arg.save_dir +
      "\n arch = " + in_arg.arch + "\n learning_rate = " + in_arg.learning_rate +
      " \n hidden_units = " + in_arg.hidden_units +  "\n eppochs = " + in_arg.eppochs + "\n device type = " + in_arg.device)

# Define training and testing folders
train_dir = 'ImageClassifier/' + in_arg.dir + 'train'
valid_dir = 'ImageClassifier/' + in_arg.dir + 'valid'

# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.Resize(224),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

data_transforms = transforms.Compose([transforms.Resize(224),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)

class_names = train_data.classes


# Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# Model definition based on command line argument
if in_arg.arch == "densenet121":
    model = models.densenet121(pretrained = True)
elif in_arg.arch == "vgg13":
    model = models.vgg13(pretrained = True)
else:
    print("Model " + in_arg.arch + " not available")
    exit()

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
# Define classifier based on command line argument
hidden_units = int(in_arg.hidden_units)
model.classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

# Define Criterion and Optimizer based on command arguments
learning_rate = float(in_arg.learning_rate)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Use GPU if it's available and based on command argument
#if in_arg.device == "GPU" and torch.cuda.is_available():
#    device = torch.device("cuda")
#else:
#    device = torch.device("cpu")
#model.to(device)
model.to("cuda")

# Run and Train the model with epochs based on command line argument
with active_session():
    
    epochs = int(in_arg.eppochs)
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        Time = time.time()
        for inputs, labels in train_loader:
            model.train()
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Step {steps}.. "
                  f"Time: {round(time.time() - Time,2)} seconds.. "
                  f"Train loss: {running_loss/len(train_loader):.3f}.. "
                  f"Test loss: {test_loss/len(valid_loader):.3f}.. "
                  f"Test accuracy: {round((accuracy.item()/len(valid_loader))*100,2):.3f} %.. ") 
            running_loss = 0

#Save the checkpoint .
now = time.localtime(time.time())
checkpoint_dir = 'ImageClassifier/' + in_arg.save_dir
time_stamp = time.strftime("%d%m%y_%H%M", now)
checkpoint_file = 'CheckFile_'+ time_stamp+'.pth'

model.class_to_idx = train_data.class_to_idx
torch.save({'arch': in_arg.arch,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            checkpoint_dir + checkpoint_file)