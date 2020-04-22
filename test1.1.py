"""
Resumer : 
Resaux de neurone de base

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
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
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
        ajoutPoidsConvolution(self, 3, 4, 3, totalSize)
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
        ajoutPoidsConvolution(self, 4, 6, 3, totalSize)
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
        ajoutPoidsConvolution(self, 6, 16, 5, totalSize)
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

        x = x.view(-1, 16 * 5 * 5)
        
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
Resulat 1 :

Files already downloaded and verified
Files already downloaded and verified
cuda:0
 frog   dog  bird   car
Accuracy of the network on the 10000 test images: 9 %
complexite=7882471728.0
complexite=11034199728.0
complexite=14185927728.0
complexite=17337655728.0
complexite=20489383728.0
complexite=23641111728.0
complexite=26792839728.0
complexite=29944567728.0
complexite=33096295728.0
complexite=36248023728.0
complexite=39399751728.0
complexite=42551479728.0
complexite=45703207728.0
evaluation after epoch 1
Accuracy of the network on the 10000 test images: 44 %
complexite=55158391728.0
complexite=58310119728.0
complexite=61461847728.0
complexite=64613575728.0
complexite=67765303728.0
complexite=70917031728.0
complexite=74068759728.0
complexite=77220487728.0
complexite=80372215728.0
complexite=83523943728.0
complexite=86675671728.0
complexite=89827399728.0
complexite=92979127728.0
evaluation after epoch 2
Accuracy of the network on the 10000 test images: 52 %
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:    cat plane  ship plane
complexiteImage=7882471728.0
complexiteEpoque=102434311728.0
complexiteTotale=196986151728.0
Accuracy of the network on the 10000 test images: 52 %
Accuracy of plane : 58 %
Accuracy of   car : 76 %
Accuracy of  bird : 48 %
Accuracy of   cat : 24 %
Accuracy of  deer : 49 %
Accuracy of   dog : 37 %
Accuracy of  frog : 71 %
Accuracy of horse : 57 %
Accuracy of  ship : 63 %
Accuracy of truck : 34 %



Resultat 2:

Files already downloaded and verified
Files already downloaded and verified
cuda:0
horse  deer   car  bird
Accuracy of the network on the 10000 test images: 10 %
complexite=7882471728.0
complexite=11034199728.0
complexite=14185927728.0
complexite=17337655728.0
complexite=20489383728.0
complexite=23641111728.0
complexite=26792839728.0
complexite=29944567728.0
complexite=33096295728.0
complexite=36248023728.0
complexite=39399751728.0
complexite=42551479728.0
complexite=45703207728.0
evaluation after epoch 1
Accuracy of the network on the 10000 test images: 44 %
complexite=55158391728.0
complexite=58310119728.0
complexite=61461847728.0
complexite=64613575728.0
complexite=67765303728.0
complexite=70917031728.0
complexite=74068759728.0
complexite=77220487728.0
complexite=80372215728.0
complexite=83523943728.0
complexite=86675671728.0
complexite=89827399728.0
complexite=92979127728.0
evaluation after epoch 2
Accuracy of the network on the 10000 test images: 48 %
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:    cat plane plane plane
complexiteImage=7882471728.0
complexiteEpoque=102434311728.0
complexiteTotale=196986151728.0
Accuracy of the network on the 10000 test images: 48 %
Accuracy of plane : 66 %
Accuracy of   car : 64 %
Accuracy of  bird : 46 %
Accuracy of   cat : 25 %
Accuracy of  deer : 49 %
Accuracy of   dog : 19 %
Accuracy of  frog : 64 %
Accuracy of horse : 60 %
Accuracy of  ship : 49 %
Accuracy of truck : 36 %




resultat 6 epoch :

Files already downloaded and verified
Files already downloaded and verified
cuda:0
plane   car  bird   cat
Accuracy of the network on the 10000 test images: 10 %
complexite=7882471728.0
complexite=11034199728.0
complexite=14185927728.0
complexite=17337655728.0
complexite=20489383728.0
complexite=23641111728.0
complexite=26792839728.0
complexite=29944567728.0
complexite=33096295728.0
complexite=36248023728.0
complexite=39399751728.0
complexite=42551479728.0
complexite=45703207728.0
evaluation after epoch 1
Accuracy of the network on the 10000 test images: 42 %
complexite=55158391728.0
complexite=58310119728.0
complexite=61461847728.0
complexite=64613575728.0
complexite=67765303728.0
complexite=70917031728.0
complexite=74068759728.0
complexite=77220487728.0
complexite=80372215728.0
complexite=83523943728.0
complexite=86675671728.0
complexite=89827399728.0
complexite=92979127728.0
evaluation after epoch 2
Accuracy of the network on the 10000 test images: 52 %
complexite=102434311728.0
complexite=105586039728.0
complexite=108737767728.0
complexite=111889495728.0
complexite=115041223728.0
complexite=118192951728.0
complexite=121344679728.0
complexite=124496407728.0
complexite=127648135728.0
complexite=130799863728.0
complexite=133951591728.0
complexite=137103319728.0
complexite=140255047728.0
evaluation after epoch 3
Accuracy of the network on the 10000 test images: 59 %
complexite=149710231728.0
complexite=152861959728.0
complexite=156013687728.0
complexite=159165415728.0
complexite=162317143728.0
complexite=165468871728.0
complexite=168620599728.0
complexite=171772327728.0
complexite=174924055728.0
complexite=178075783728.0
complexite=181227511728.0
complexite=184379239728.0
complexite=187530967728.0
evaluation after epoch 4
Accuracy of the network on the 10000 test images: 61 %
complexite=196986151728.0
complexite=200137879728.0
complexite=203289607728.0
complexite=206441335728.0
complexite=209593063728.0
complexite=212744791728.0
complexite=215896519728.0
complexite=219048247728.0
complexite=222199975728.0
complexite=225351703728.0
complexite=228503431728.0
complexite=231655159728.0
complexite=234806887728.0
evaluation after epoch 5
Accuracy of the network on the 10000 test images: 62 %
complexite=244262071728.0
complexite=247413799728.0
complexite=250565527728.0
complexite=253717255728.0
complexite=256868983728.0
complexite=260020711728.0
complexite=263172439728.0
complexite=266324167728.0
complexite=269475895728.0
complexite=272627623728.0
complexite=275779351728.0
complexite=278931079728.0
complexite=282082807728.0
evaluation after epoch 6
Accuracy of the network on the 10000 test images: 62 %
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:   ship  ship  ship plane
complexiteImage=7882471728.0
complexiteEpoque=102434311728.0
complexiteTotale=575193511728.0
Accuracy of the network on the 10000 test images: 62 %
Accuracy of plane : 59 %
Accuracy of   car : 88 %
Accuracy of  bird : 46 %
Accuracy of   cat : 48 %
Accuracy of  deer : 48 %
Accuracy of   dog : 47 %
Accuracy of  frog : 72 %
Accuracy of horse : 66 %
Accuracy of  ship : 80 %
Accuracy of truck : 60 %


reultat epoch 6 et operation par seconde

Files already downloaded and verified
Files already downloaded and verified
cuda:0
 frog   car truck plane
Accuracy of the network on the 10000 test images: 10 %
complexite=7882471728.0
complexite=11034199728.0
complexite=14185927728.0
complexite=17337655728.0
complexite=20489383728.0
complexite=23641111728.0
complexite=26792839728.0
complexite=29944567728.0
complexite=33096295728.0
complexite=36248023728.0
complexite=39399751728.0
complexite=42551479728.0
complexite=45703207728.0
evaluation after epoch 1
Accuracy of the network on the 10000 test images: 41 %
complexite=55158391728.0
complexite=58310119728.0
complexite=61461847728.0
complexite=64613575728.0
complexite=67765303728.0
complexite=70917031728.0
complexite=74068759728.0
complexite=77220487728.0
complexite=80372215728.0
complexite=83523943728.0
complexite=86675671728.0
complexite=89827399728.0
complexite=92979127728.0
evaluation after epoch 2
Accuracy of the network on the 10000 test images: 49 %
complexite=102434311728.0
complexite=105586039728.0
complexite=108737767728.0
complexite=111889495728.0
complexite=115041223728.0
complexite=118192951728.0
complexite=121344679728.0
complexite=124496407728.0
complexite=127648135728.0
complexite=130799863728.0
complexite=133951591728.0
complexite=137103319728.0
complexite=140255047728.0
evaluation after epoch 3
Accuracy of the network on the 10000 test images: 54 %
complexite=149710231728.0
complexite=152861959728.0
complexite=156013687728.0
complexite=159165415728.0
complexite=162317143728.0
complexite=165468871728.0
complexite=168620599728.0
complexite=171772327728.0
complexite=174924055728.0
complexite=178075783728.0
complexite=181227511728.0
complexite=184379239728.0
complexite=187530967728.0
evaluation after epoch 4
Accuracy of the network on the 10000 test images: 58 %
complexite=196986151728.0
complexite=200137879728.0
complexite=203289607728.0
complexite=206441335728.0
complexite=209593063728.0
complexite=212744791728.0
complexite=215896519728.0
complexite=219048247728.0
complexite=222199975728.0
complexite=225351703728.0
complexite=228503431728.0
complexite=231655159728.0
complexite=234806887728.0
evaluation after epoch 5
Accuracy of the network on the 10000 test images: 60 %
complexite=244262071728.0
complexite=247413799728.0
complexite=250565527728.0
complexite=253717255728.0
complexite=256868983728.0
complexite=260020711728.0
complexite=263172439728.0
complexite=266324167728.0
complexite=269475895728.0
complexite=272627623728.0
complexite=275779351728.0
complexite=278931079728.0
complexite=282082807728.0
evaluation after epoch 6
Accuracy of the network on the 10000 test images: 61 %
Temps d execution : 282.984304189682 secondes ---
282.98440647125244
2004742692.575265
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:    cat   car  ship plane
complexiteImage=7882471728.0
complexiteEpoque=102434311728.0
complexiteTotale=575193511728.0
Accuracy of the network on the 10000 test images: 61 %
Accuracy of plane : 73 %
Accuracy of   car : 81 %
Accuracy of  bird : 53 %
Accuracy of   cat : 29 %
Accuracy of  deer : 67 %
Accuracy of   dog : 33 %
Accuracy of  frog : 77 %
Accuracy of horse : 62 %
Accuracy of  ship : 61 %
Accuracy of truck : 74 %
Out[11]: '\nResulat 1 :\n\nFiles already downloaded and verified\nFiles already downloaded and verified\ncuda:0\n frog   dog  bird   car\nAccuracy of the network on the 10000 test images: 9 %\ncomplexite=7882471728.0\ncomplexite=11034199728.0\ncomplexite=14185927728.0\ncomplexite=17337655728.0\ncomplexite=20489383728.0\ncomplexite=23641111728.0\ncomplexite=26792839728.0\ncomplexite=29944567728.0\ncomplexite=33096295728.0\ncomplexite=36248023728.0\ncomplexite=39399751728.0\ncomplexite=42551479728.0\ncomplexite=45703207728.0\nevaluation after epoch 1\nAccuracy of the network on the 10000 test images: 44 %\ncomplexite=55158391728.0\ncomplexite=58310119728.0\ncomplexite=61461847728.0\ncomplexite=64613575728.0\ncomplexite=67765303728.0\ncomplexite=70917031728.0\ncomplexite=74068759728.0\ncomplexite=77220487728.0\ncomplexite=80372215728.0\ncomplexite=83523943728.0\ncomplexite=86675671728.0\ncomplexite=89827399728.0\ncomplexite=92979127728.0\nevaluation after epoch 2\nAccuracy of the network on the 10000 test images: 52 %\nFinished Training\nGroundTruth:    cat  ship  ship plane\nPredicted:    cat plane  ship plane\ncomplexiteImage=7882471728.0\ncomplexiteEpoque=102434311728.0\ncomplexiteTotale=196986151728.0\nAccuracy of the network on the 10000 test images: 52 %\nAccuracy of plane : 58 %\nAccuracy of   car : 76 %\nAccuracy of  bird : 48 %\nAccuracy of   cat : 24 %\nAccuracy of  deer : 49 %\nAccuracy of   dog : 37 %\nAccuracy of  frog : 71 %\nAccuracy of horse : 57 %\nAccuracy of  ship : 63 %\nAccuracy of truck : 34 %\n\n\n\nResultat 2:\n\nFiles already downloaded and verified\nFiles already downloaded and verified\ncuda:0\nhorse  deer   car  bird\nAccuracy of the network on the 10000 test images: 10 %\ncomplexite=7882471728.0\ncomplexite=11034199728.0\ncomplexite=14185927728.0\ncomplexite=17337655728.0\ncomplexite=20489383728.0\ncomplexite=23641111728.0\ncomplexite=26792839728.0\ncomplexite=29944567728.0\ncomplexite=33096295728.0\ncomplexite=36248023728.0\ncomplexite=39399751728.0\ncomplexite=42551479728.0\ncomplexite=45703207728.0\nevaluation after epoch 1\nAccuracy of the network on the 10000 test images: 44 %\ncomplexite=55158391728.0\ncomplexite=58310119728.0\ncomplexite=61461847728.0\ncomplexite=64613575728.0\ncomplexite=67765303728.0\ncomplexite=70917031728.0\ncomplexite=74068759728.0\ncomplexite=77220487728.0\ncomplexite=80372215728.0\ncomplexite=83523943728.0\ncomplexite=86675671728.0\ncomplexite=89827399728.0\ncomplexite=92979127728.0\nevaluation after epoch 2\nAccuracy of the network on the 10000 test images: 48 %\nFinished Training\nGroundTruth:    cat  ship  ship plane\nPredicted:    cat plane plane plane\ncomplexiteImage=7882471728.0\ncomplexiteEpoque=102434311728.0\ncomplexiteTotale=196986151728.0\nAccuracy of the network on the 10000 test images: 48 %\nAccuracy of plane : 66 %\nAccuracy of   car : 64 %\nAccuracy of  bird : 46 %\nAccuracy of   cat : 25 %\nAccuracy of  deer : 49 %\nAccuracy of   dog : 19 %\nAccuracy of  frog : 64 %\nAccuracy of horse : 60 %\nAccuracy of  ship : 49 %\nAccuracy of truck : 36 %\n\n\n\n\nresultat 6 epoch :\n\nFiles already downloaded and verified\nFiles already downloaded and verified\ncuda:0\nplane   car  bird   cat\nAccuracy of the network on the 10000 test images: 10 %\ncomplexite=7882471728.0\ncomplexite=11034199728.0\ncomplexite=14185927728.0\ncomplexite=17337655728.0\ncomplexite=20489383728.0\ncomplexite=23641111728.0\ncomplexite=26792839728.0\ncomplexite=29944567728.0\ncomplexite=33096295728.0\ncomplexite=36248023728.0\ncomplexite=39399751728.0\ncomplexite=42551479728.0\ncomplexite=45703207728.0\nevaluation after epoch 1\nAccuracy of the network on the 10000 test images: 42 %\ncomplexite=55158391728.0\ncomplexite=58310119728.0\ncomplexite=61461847728.0\ncomplexite=64613575728.0\ncomplexite=67765303728.0\ncomplexite=70917031728.0\ncomplexite=74068759728.0\ncomplexite=77220487728.0\ncomplexite=80372215728.0\ncomplexite=83523943728.0\ncomplexite=86675671728.0\ncomplexite=89827399728.0\ncomplexite=92979127728.0\nevaluation after epoch 2\nAccuracy of the network on the 10000 test images: 52 %\ncomplexite=102434311728.0\ncomplexite=105586039728.0\ncomplexite=108737767728.0\ncomplexite=111889495728.0\ncomplexite=115041223728.0\ncomplexite=118192951728.0\ncomplexite=121344679728.0\ncomplexite=124496407728.0\ncomplexite=127648135728.0\ncomplexite=130799863728.0\ncomplexite=133951591728.0\ncomplexite=137103319728.0\ncomplexite=140255047728.0\nevaluation after epoch 3\nAccuracy of the network on the 10000 test images: 59 %\ncomplexite=149710231728.0\ncomplexite=152861959728.0\ncomplexite=156013687728.0\ncomplexite=159165415728.0\ncomplexite=162317143728.0\ncomplexite=165468871728.0\ncomplexite=168620599728.0\ncomplexite=171772327728.0\ncomplexite=174924055728.0\ncomplexite=178075783728.0\ncomplexite=181227511728.0\ncomplexite=184379239728.0\ncomplexite=187530967728.0\nevaluation after epoch 4\nAccuracy of the network on the 10000 test images: 61 %\ncomplexite=196986151728.0\ncomplexite=200137879728.0\ncomplexite=203289607728.0\ncomplexite=206441335728.0\ncomplexite=209593063728.0\ncomplexite=212744791728.0\ncomplexite=215896519728.0\ncomplexite=219048247728.0\ncomplexite=222199975728.0\ncomplexite=225351703728.0\ncomplexite=228503431728.0\ncomplexite=231655159728.0\ncomplexite=234806887728.0\nevaluation after epoch 5\nAccuracy of the network on the 10000 test images: 62 %\ncomplexite=244262071728.0\ncomplexite=247413799728.0\ncomplexite=250565527728.0\ncomplexite=253717255728.0\ncomplexite=256868983728.0\ncomplexite=260020711728.0\ncomplexite=263172439728.0\ncomplexite=266324167728.0\ncomplexite=269475895728.0\ncomplexite=272627623728.0\ncomplexite=275779351728.0\ncomplexite=278931079728.0\ncomplexite=282082807728.0\nevaluation after epoch 6\nAccuracy of the network on the 10000 test images: 62 %\nFinished Training\nGroundTruth:    cat  ship  ship plane\nPredicted:   ship  ship  ship plane\ncomplexiteImage=7882471728.0\ncomplexiteEpoque=102434311728.0\ncomplexiteTotale=575193511728.0\nAccuracy of the network on the 10000 test images: 62 %\nAccuracy of plane : 59 %\nAccuracy of   car : 88 %\nAccuracy of  bird : 46 %\nAccuracy of   cat : 48 %\nAccuracy of  deer : 48 %\nAccuracy of   dog : 47 %\nAccuracy of  frog : 72 %\nAccuracy of horse : 66 %\nAccuracy of  ship : 80 %\nAccuracy of truck : 60 %\n\n\n'


"""