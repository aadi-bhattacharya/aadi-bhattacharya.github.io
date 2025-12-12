+++
title = "DaSHLabs, ML-sys :: 1-1"
date = "2025-11-20"
+++

time period: 3rd Nov 2025 -- 11th Nov 2025

### intro and conclusion
i learned an enormous amount, in a relatively short period of time for DaSH Labs' 2025 intake in the systems for ML subsystem.
from feedforward networks, to attention mechanisms, i enjoyed learning something on my own.

i implemented a federated average neural network for the MNIST database, using pytorch.

unfortunately, i did not make it to the interview round, but it was a great learning experience for me.

## assignment 1.

> i might be wrong but i think the reason i didn't make it to interviews was i accidentally updated my google form to an unfinished draft(?) [i have contacted a PoR about it.]

- [refer to the quiz here](https://github.com/aadi-bhattacharya/dashlabs-1-1/tree/main/task_1)

#### notes on assignment 1:
- lorem
- ipsum
- dot
- amet

A1:
```
Alternatives to ReLU: tanh function, sigmoid function.
Need for non-linearity >>
I think non-linearity is needed because everything can't be expressed in linear functions,
By everything i mean different situations. Situations aren't always simple, and while "A" may imply "B", "C" might not imply "D".
ReLU is the simplest(? I think it's the simplest because it literally is just max(0,x)) non-linear function. Hence we aren't spending a lot of compute in ReLU, and removing it would lose non-linearity together making the entire function a single linear expression, losing complexity and making it not useful.
```
A2:
```
See, quantization in this case involves the mapping of FP16 to INT8, which obviously leads to the loss of precision in weights and biases. Like the matrix [-0.9, -0.1, 0.0, 0.3, 0.7, 1.0] gets converted to [0, 108, 121, 161, 215, 255] approximately. Here the scaling factor will be 1.9/255 with a zero point approximately equal to 121.

Since quantization causes a loss in precision, involves clipping, weights and biases may not be "proper" enough to generate useful enough results.
If the calibration is not proper, we will have even poorer results.
Errors may stack together over the layers to become even worse.
```
A3:
```
Since prompts are usually very diverse, we need to use alot of attention heads instead of a single large attention head so that the different attention heads can focus on different properties or features of the prompt. A single large attention head may be able to capture the data, but using multiple attention heads adds depth to the representation/understanding of the input prompt. With multiple attention heads, the model will also not try to overfit to a single pattern in the prompt.
```
A4:
```
The main issue is Pranav setting the learning rate value to be 1, thinking that it will help speed up the learning algorithm, but instead of that happening, what happens is the value oscillates due to it overshooting each time it tries to calculate the backpropogation to optimise the value.
If we think of it as the surface of a curve, the learning rate cause it jump around the orifice of the valley and never being precise enough to converge and settle to a single point.
```
A5:
```
Adding more neurons is causing the neural network to work specifically for the training set and making it worse at the unknown general validation set. Increasing the number of neurons helps increase capacity to fit the learning set too much.

To mitigate this we can increase the number of training images but since we are pulling the images from the standard MNIST library perhaps its not the way to go about it.
It might be useful to maybe simplify/dumbdown the nn a bit by decreasing the number of neurons in the layers.

There are other useful methods too like randomly turning off few neurons to make sure that a specific pattern is not being memorised.
```
A6:
```
Parallelisation can be done with data parallelism, model parallelism, and pipeline parallelism.
Data parallelism is basically the splitting of the training data into different batches, with each GPU having a full copy of the model, and the gradient getting averaged to keep them all in sync.
Model parallelism is the splitting of the model into either intra layer parallelism, or inter layer parallelism. Intra-layer parallelism makes the different GPUs do different multiplications in the same layer, whereas inter layer parallelism makes different GPUs do different layers.
Pipeline parallelism is basically a specialised form of interlayer model parallelism, where the training batch is split into microbatches and then basically interlayer parallelism takes place. There are methods to make this even faster like 1F1B
```
A7:
```
Batching is basically collecting inputs until the batch size is fulfilled and then running it on the neural network.
Having a batch size of 64 may be too large for DaSHGPT because it just has around 70 members using it. The batch size being 64 causes the first users to wait for the last users.. Hence this increases the waiting time by quite a considerable amount.
An easy fix to this would be statically decreasing the batch size to say 8, or maybe implementing a larger batchsize based on the number of users available (connected).
There also is dynamic batching which lets the system choose between running the nn every set amount of time or collecting and running the filled batch, whichever is prior to come.
```
A8:
```
"Repeating this procedure from the beginning each time a new token is added", meaning that the model is not using the previous hidden states. The model is redoing work that has already been done leading to more latency.
```
A9:
```
Atharva is making a mistake by letting the parametric and context information from prior prompts pile up in the VRAM, since each and every prompt is using context from the prior one, the activations and hidden states that were generated for the prior promt is sotred in the VRAM. Everytime he runs a prompt it ends up increasing the memory usage further.
We need to somehow free these intermediate expressions, effectively enough so that it retains the necessary context and discards the unneeded info.
We can also move the data from VRAM to RAM.
```
A10:
```
As per the hint, if we store the KV cache in RAM instead of VRAM it should be the cause of the delay.
To fix it, we shouldn't move it into the RAM after each prompt/token generisation. We can also limit the how much of it we store, storing lesser.
```

## assignment 2.
```py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class mModel(nn.Module):
    def __init__(s):
        nn.Module.__init__(s)
        s.layer1 = nn.Linear(28*28, 64)
        s.layer2 = nn.Linear(64, 10)

    def forward(s, x):
        x = x.view(-1, 28*28)
        x = nn.functional.relu(s.layer1(x))
        x = s.layer2(x)
        return nn.functional.log_softmax(x, dim=1)

def trainingc(model, data_loader, lr=0.1, epochs=2):
    opt = optim.SGD(model.parameters(), lr)
    model.train()
    for e in range(epochs):
        for imgs, labels in data_loader:
            opt.zero_grad()
            out = model(imgs)
            loss = nn.functional.nll_loss(out, labels)
            loss.backward()
            opt.step()
    return model.state_dict()

def average(models):
    avg = {}
    for i in models[0]:
        s = 0
        for m in models:
            s +=m[i]
        avg[i] = s / len(models)
    return avg

def check_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    print("Accuracy:", 100 * correct / total, "%")

def federated():
    trans = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST('.', train=True, download=True, transform=trans)
    test = datasets.MNIST('.', train=False, transform=trans)
    loader = DataLoader(test, batch_size=64, shuffle=False)

    nclients = 2
    size = len(train) // nclients
    parts = random_split(train, [size]*nclients)
    loaders = [DataLoader(p, batch_size=32, shuffle=True) for p in parts]

    globalm = mModel()

    for itr in range(2):
        print("iteration", itr+1)
        new_weights = []
        for i in range(nclients):
            print("client",i+1,"training")
            localm = mModel()
            localm.load_state_dict(globalm.state_dict())
            trained = trainingc(localm, loaders[i])
            new_weights.append(trained)
        globalm.load_state_dict(average(new_weights))
    check_accuracy(globalm, loader)

federated()
```
