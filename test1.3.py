"""
Resumer : 
conv1:3-6-3
conv2:6-8-3
conv3:8-16-5
conv4:16-32-5

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
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 1 * 1, 120)
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



        #Conv4
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsConvolution(self, 16, 32, 6, totalSize)
        x = self.conv4(x)
        if (debogage == 1):
            print("Conv 4")
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

        

        x = x.view(-1, 32 * 1 * 1)
        
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
Resultat :

Files already downloaded and verified
Files already downloaded and verified
cpu
 deer  deer   cat  bird
Accuracy of the network on the 10000 test images: 10 %
complexite=12492635056.0
complexite=17487691056.0
complexite=22482747056.0
complexite=27477803056.0
complexite=32472859056.0
complexite=37467915056.0
complexite=42462971056.0
complexite=47458027056.0
complexite=52453083056.0
complexite=57448139056.0
complexite=62443195056.0
complexite=67438251056.0
complexite=72433307056.0
evaluation after epoch 1
Accuracy of the network on the 10000 test images: 33 %
complexite=87418475056.0
complexite=92413531056.0
complexite=97408587056.0
complexite=102403643056.0
complexite=107398699056.0
complexite=112393755056.0
complexite=117388811056.0
complexite=122383867056.0
complexite=127378923056.0
complexite=132373979056.0
complexite=137369035056.0
complexite=142364091056.0
complexite=147359147056.0
evaluation after epoch 2
Accuracy of the network on the 10000 test images: 45 %
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:    cat   car  ship  ship
complexiteImage=12492635056.0
complexiteEpoque=162344315056.0
complexiteTotale=312195995056.0
Accuracy of the network on the 10000 test images: 45 %
Accuracy of plane : 58 %
Accuracy of   car : 64 %
Accuracy of  bird : 15 %
Accuracy of   cat : 23 %
Accuracy of  deer : 50 %
Accuracy of   dog : 25 %
Accuracy of  frog : 63 %
Accuracy of horse : 55 %
Accuracy of  ship : 63 %
Accuracy of truck : 38 %

test 6 epoch et time

Files already downloaded and verified
Files already downloaded and verified
cuda:0
plane  bird  bird  ship
Accuracy of the network on the 10000 test images: 10 %
complexite=12492635056.0
complexite=17487691056.0
complexite=22482747056.0
complexite=27477803056.0
complexite=32472859056.0
complexite=37467915056.0
complexite=42462971056.0
complexite=47458027056.0
complexite=52453083056.0
complexite=57448139056.0
complexite=62443195056.0
complexite=67438251056.0
complexite=72433307056.0
evaluation after epoch 1
Accuracy of the network on the 10000 test images: 37 %
complexite=87418475056.0
complexite=92413531056.0
complexite=97408587056.0
complexite=102403643056.0
complexite=107398699056.0
complexite=112393755056.0
complexite=117388811056.0
complexite=122383867056.0
complexite=127378923056.0
complexite=132373979056.0
complexite=137369035056.0
complexite=142364091056.0
complexite=147359147056.0
evaluation after epoch 2
Accuracy of the network on the 10000 test images: 45 %
complexite=162344315056.0
complexite=167339371056.0
complexite=172334427056.0
complexite=177329483056.0
complexite=182324539056.0
complexite=187319595056.0
complexite=192314651056.0
complexite=197309707056.0
complexite=202304763056.0
complexite=207299819056.0
complexite=212294875056.0
complexite=217289931056.0
complexite=222284987056.0
evaluation after epoch 3
Accuracy of the network on the 10000 test images: 51 %
complexite=237270155056.0
complexite=242265211056.0
complexite=247260267056.0
complexite=252255323056.0
complexite=257250379056.0
complexite=262245435056.0
complexite=267240491056.0
complexite=272235547056.0
complexite=277230603056.0
complexite=282225659056.0
complexite=287220715056.0
complexite=292215771056.0
complexite=297210827056.0
evaluation after epoch 4
Accuracy of the network on the 10000 test images: 50 %
complexite=312195995056.0
complexite=317191051056.0
complexite=322186107056.0
complexite=327181163056.0
complexite=332176219056.0
complexite=337171275056.0
complexite=342166331056.0
complexite=347161387056.0
complexite=352156443056.0
complexite=357151499056.0
complexite=362146555056.0
complexite=367141611056.0
complexite=372136667056.0
evaluation after epoch 5
Accuracy of the network on the 10000 test images: 55 %
complexite=387121835056.0
complexite=392116891056.0
complexite=397111947056.0
complexite=402107003056.0
complexite=407102059056.0
complexite=412097115056.0
complexite=417092171056.0
complexite=422087227056.0
complexite=427082283056.0
complexite=432077339056.0
complexite=437072395056.0
complexite=442067451056.0
complexite=447062507056.0
evaluation after epoch 6
Accuracy of the network on the 10000 test images: 56 %
Temps d execution : 298.58680510520935 secondes ---
298.58689427375793
3011216714.181709
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:    cat  ship  ship  ship
complexiteImage=12492635056.0
complexiteEpoque=162344315056.0
complexiteTotale=911602715056.0
Accuracy of the network on the 10000 test images: 56 %
Accuracy of plane : 51 %
Accuracy of   car : 64 %
Accuracy of  bird : 53 %
Accuracy of   cat : 48 %
Accuracy of  deer : 49 %
Accuracy of   dog : 38 %
Accuracy of  frog : 53 %
Accuracy of horse : 66 %
Accuracy of  ship : 83 %
Accuracy of truck : 56 %


"""