"""
Resumer : 
conv1:4-8-3
conv2:8-16-3
conv3:16-64-5

Critère d'évaluation : CrossEntropyLoss

"""


##1. Loading and normalizing CIFAR10

import torch
import torchvision
import torchvision.transforms as transforms
import time
##The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1]. .. note:

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Définition variables complexité
complexiteImage = 0
complexiteEpoque = 0
complexiteTotale = 0

#Variable affichage debogage
debogage = 0

### Utilisation GPU CUDA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

### Fin utilisation GPU CUDA

##Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))





##2. Define a Convolutional Neural Network

##Copy the neural network from the Neural Networks section before and modify it to take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    complexite = 0

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def getComplexite(self):
        return self.complexite

    def forward(self, x):
        #print(x)
        if (debogage == 1):
            print(x.size())
            print("Conv 1")
        #Conv 1
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsConvolution(self, 4, 8, 3, totalSize)
        x = self.conv1(x)
        if (debogage == 1):
            print(x.size())
            print("Relu")
        #Relu
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsRelu(self, totalSize)
        x = F.relu(x)
        if (debogage == 1):
            print(x.size())
            print("Conv 2")
        #Conv2
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsConvolution(self, 8, 16, 3, totalSize)
        x = self.conv2(x)
        if (debogage == 1):
            print(x.size())
            print("Relu")
        #Relu
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsRelu(self, totalSize)
        x = F.relu(x)
        if (debogage == 1):
            print(x.size())
            print("Pool")
        #Pool
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsMaxPool(self, 2, totalSize)
        x = self.pool(x)
        if (debogage == 1):
            print(x.size())
            print("Conv 3")
        #Conv3
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsConvolution(self, 16, 64, 5, totalSize)
        x = self.conv3(x)
        if (debogage == 1):
            print(x.size())
            print("Relu")
        #Relu
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsRelu(self, totalSize)
        x = F.relu(x)
        if (debogage == 1):
            print(x.size())
            print("Pool")
        #Pool
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsMaxPool(self, 2, totalSize)
        x = self.pool(x)

        if (debogage == 1):
            print(x.size())

        x = x.view(-1, 64 * 5 * 5)
        
        #Linéaire1
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]
        x = self.fc1(x)
        ajoutPoidsLinear(self, x)
        #Relu
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]
        ajoutPoidsRelu(self, totalSize)
        x = F.relu(x)
        #Linéaire2
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]
        x = self.fc2(x)
        ajoutPoidsLinear(self, x)
        #Relu
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]
        ajoutPoidsRelu(self, totalSize)
        x = F.relu(x)
        #Linéaire3
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]
        x = self.fc3(x)
        ajoutPoidsLinear(self, x)
        #print ("complexite ="+str(self.complexite))
        return x


net = Net() 
net.to(device)



##3. Define a Loss function and optimizer

##Let’s use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

##4. Train the network

##Fonction d'évalutaion (Partie Evaluation en continu du système):
def evaluation():
	correct = 0
	total = 0
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data[0].to(device), data[1].to(device)
	        outputs = net(images)
	        _, predicted = torch.max(outputs.data, 1)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
	    100 * correct / total))

"""Ajout au calcul de la complexité totale 
la complexité d'un relu sur un tenseur de taille size """
def ajoutPoidsRelu(self, size):
    self.complexite += size

"""Ajout au calcul de la complexité totale 
la complexité d'un convolution(x,y,k), sur un tenseur de taille size"""
def ajoutPoidsConvolution(self, x, y, k, size):
    compKernel = k*k
    self.complexite += compKernel * size * y

"""Ajout au calcul de la complexité totale 
la complexité d'un MaxPool2D(k, s), sur un tenseur de taille size"""
def ajoutPoidsMaxPool(self, k, size):
    self.complexite += size / (k*k)

"""Ajout au calcul de la complexité totale 
la complexité d'un MaxPool2D(k, s), sur un tenseur de taille size"""
def ajoutPoidsLinear(self, y):
    self.complexite += 4 * (y.size()[0] * y.size()[1])


##This is when things start to get interesting. We simply have to loop over our data iterator, and feed the inputs to the network and optimize.

evaluation()
# Debut du decompte du temps
start_time = time.time()
for epoch  in range(6):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        if(epoch == 0 and i == 0):
            complexiteImage = net.getComplexite()
        if(i%1000 == 0):
            print("complexite="+str(net.getComplexite()))
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if(epoch == 0):
        complexiteEpoque = net.getComplexite()*2 #Multiplié par 2 pour compter passes avant et arrière

    complexiteTotale = net.getComplexite()*2 #Multiplié par 2 pour compter passes avant et arrière

    print('evaluation after epoch', (epoch + 1))
    evaluation()

# Affichage du temps d execution
print("Temps d execution : %s secondes ---" % (time.time() - start_time))
print( time.time() - start_time )
print( complexiteTotale / (time.time() - start_time) )

print('Finished Training')

#Let’s quickly save our trained model:

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

##5. Test the network on the test data

##We have trained the network for 2 passes over the training dataset. But we need to check if the network has learnt anything at all.

##We will check this by predicting the class label that the neural network outputs, and checking it against the ground-truth. If the prediction is correct, we add the sample to the list of correct predictions.

##Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

##Next, let’s load back in our saved model (note: saving and re-loading the model wasn’t necessary here, we only did it to illustrate how to do so):

net = Net()
net.load_state_dict(torch.load(PATH))

##Okay, now let us see what the neural network thinks these examples above are:

outputs = net(images)

##The outputs are energies for the 10 classes. The higher the energy for a class, the more the network thinks that the image is of the particular class. So, let’s get the index of the highest energy:

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

##The results seem pretty good.

##Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    #Ajout des passes avant du test à la complexité
    complexiteEpoque += net.getComplexite()
    complexiteTotale += net.getComplexite()

print("complexiteImage="+str(complexiteImage))
print("complexiteEpoque="+str(complexiteEpoque))
print("complexiteTotale="+str(complexiteTotale))

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


##That looks way better than chance, which is 10% accuracy (randomly picking a class out of 10 classes). Seems like the network learnt something.

##Hmmm, what are the classes that performed well, and the classes that did not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))



"""
Resultat 6epoch plus time

Files already downloaded and verified
Files already downloaded and verified
cuda:0
  cat  ship   dog   car
Accuracy of the network on the 10000 test images: 11 %
complexite=63100470096.0
complexite=88330566096.0
complexite=113560662096.0
complexite=138790758096.0
complexite=164020854096.0
complexite=189250950096.0
complexite=214481046096.0
complexite=239711142096.0
complexite=264941238096.0
complexite=290171334096.0
complexite=315401430096.0
complexite=340631526096.0
complexite=365861622096.0
evaluation after epoch 1
Accuracy of the network on the 10000 test images: 51 %
complexite=441551910096.0
complexite=466782006096.0
complexite=492012102096.0
complexite=517242198096.0
complexite=542472294096.0
complexite=567702390096.0
complexite=592932486096.0
complexite=618162582096.0
complexite=643392678096.0
complexite=668622774096.0
complexite=693852870096.0
complexite=719082966096.0
complexite=744313062096.0
evaluation after epoch 2
Accuracy of the network on the 10000 test images: 59 %
complexite=820003350096.0
complexite=845233446096.0
complexite=870463542096.0
complexite=895693638096.0
complexite=920923734096.0
complexite=946153830096.0
complexite=971383926096.0
complexite=996614022096.0
complexite=1021844118096.0
complexite=1047074214096.0
complexite=1072304310096.0
complexite=1097534406096.0
complexite=1122764502096.0
evaluation after epoch 3
Accuracy of the network on the 10000 test images: 67 %
complexite=1198454790096.0
complexite=1223684886096.0
complexite=1248914982096.0
complexite=1274145078096.0
complexite=1299375174096.0
complexite=1324605270096.0
complexite=1349835366096.0
complexite=1375065462096.0
complexite=1400295558096.0
complexite=1425525654096.0
complexite=1450755750096.0
complexite=1475985846096.0
complexite=1501215942096.0
evaluation after epoch 4
Accuracy of the network on the 10000 test images: 70 %
complexite=1576906230096.0
complexite=1602136326096.0
complexite=1627366422096.0
complexite=1652596518096.0
complexite=1677826614096.0
complexite=1703056710096.0
complexite=1728286806096.0
complexite=1753516902096.0
complexite=1778746998096.0
complexite=1803977094096.0
complexite=1829207190096.0
complexite=1854437286096.0
complexite=1879667382096.0
evaluation after epoch 5
Accuracy of the network on the 10000 test images: 71 %
complexite=1955357670096.0
complexite=1980587766096.0
complexite=2005817862096.0
complexite=2031047958096.0
complexite=2056278054096.0
complexite=2081508150096.0
complexite=2106738246096.0
complexite=2131968342096.0
complexite=2157198438096.0
complexite=2182428534096.0
complexite=2207658630096.0
complexite=2232888726096.0
complexite=2258118822096.0
evaluation after epoch 6
Accuracy of the network on the 10000 test images: 71 %
Temps d execution : 272.08198523521423 secondes ---
272.08207726478577
16691346373.283804
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:   ship  ship  ship plane
complexiteImage=63100470096.0
complexiteEpoque=820003350096.0
complexiteTotale=4604517750096.0
Accuracy of the network on the 10000 test images: 71 %
Accuracy of plane : 74 %
Accuracy of   car : 85 %
Accuracy of  bird : 51 %
Accuracy of   cat : 44 %
Accuracy of  deer : 78 %
Accuracy of   dog : 60 %
Accuracy of  frog : 77 %
Accuracy of horse : 75 %
Accuracy of  ship : 88 %
Accuracy of truck : 74 %

"""