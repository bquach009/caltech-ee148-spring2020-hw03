import matplotlib.pyplot as plt 
from main import Net
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sn
import pandas as pd 

import os

def find_wrong(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    images = []
    preds = []
    true = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for i in range(len(data)):
                if pred[i] != target[i]:
                    images.append(data[i])
                    preds.append(pred[i])
                    true.append(target[i])
    return images, preds, true

def gen_preds(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    preds = []
    true = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            preds.extend(pred)
            true.extend(target)
    return preds, true

def predict(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    labels = []
    images = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            labels.extend(target)
            images.extend(data)
    return labels, images

device = torch.device("cpu")
kwargs = {}
model_path = "models/mnist_model_better.pt"
model = Net().to(device)

# Set up for getting embeddings later
global embeddings
embeddings = []
def hook_fn(module, input, output):
    global embeddings
    embeddings.extend(output.numpy())
hook = model.fc2.register_forward_hook(hook_fn)

model.load_state_dict(torch.load(model_path))

test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1000, shuffle=True, **kwargs)

images, preds, true = find_wrong(model, device, test_loader)

# # Take the first 9
images = images[:9]
preds = preds[:9]
true = true[:9]

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
# Plot the images along with their predicted label and true label 
curr = 0
for i in range(3):
    for j in range(3):
        axs[i, j].imshow(images[curr][0], cmap="gray")
        axs[i, j].set_title("Label: {} Pred: {}".format(true[curr], preds[curr][0]))
        curr += 1
plt.savefig("incorrect_preds.png")

# Generate visualization of the kernels 
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
weights = model.conv1.weight.detach().numpy()
curr = 0
for i in range(3):
    for j in range(3):
        axs[i, j].imshow(weights[curr][0], cmap="gray")
        axs[i, j].set_title("Kernel: {}".format(curr))
        curr += 1
plt.savefig("kernels.png")

# Generate the confusion matrix
preds, labels = gen_preds(model, device, test_loader)
res = confusion_matrix(labels, preds)

df_cm = pd.DataFrame(res, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.ylabel("Ground Truth")
plt.xlabel("Prediction")
plt.savefig("confusion_matrix.png")

Gather feature embeddings on test set
labels, images = predict(model, device, test_loader)

print("Embeddings Obtained! Starting TSNE!")
reduced = TSNE(n_components=2).fit_transform(embeddings)
print("TSNE Done!")
np.save("embeddings.npy", reduced)
np.save("labels.npy", labels)

sorted_embeddings_x = {}
sorted_embeddings_y = {}
for i in range(10):
    sorted_embeddings_x[i] = []
    sorted_embeddings_y[i] = []
for i, label in enumerate(labels):
    sorted_embeddings_x[label.item()].append(reduced[i][0])
    sorted_embeddings_y[label.item()].append(reduced[i][1])

print("Starting Plotting!")
for i in range(10):
    plt.scatter(sorted_embeddings_x[i], sorted_embeddings_y[i])

plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
plt.savefig("embeddings_visualized.png")

# Use 4 random images and find 8 most similar vectors.
labels, images = predict(model, device, test_loader)
indices = list(range(len(labels)))
references = np.random.choice(indices, 4, replace=False)
fig, axs = plt.subplots(4, 8, figsize=(10, 20))
curr = 0
for idx in references:
    distances = []
    ref = embeddings[idx]
    for item in embeddings:
        distances.append(np.linalg.norm(item - ref))
    indices = np.argsort(distances)
    for i in range(8):
        axs[curr, i].imshow(images[indices[i]][0], cmap = "gray")
    curr += 1
plt.savefig("closestimages.png")

