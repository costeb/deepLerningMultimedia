"""
Resumer : 
conv1:3-6-3
conv2:6-8-3
conv3:8-16-5

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
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8, 16, 5)
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
        ajoutPoidsConvolution(self, 3, 6, 3, totalSize)
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
        ajoutPoidsConvolution(self, 6, 8, 3, totalSize)
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
        ajoutPoidsConvolution(self, 8, 16, 5, totalSize)
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
Resultat 1:

Files already downloaded and verified
Files already downloaded and verified
cuda:0
 bird plane   cat   cat
Accuracy of the network on the 10000 test images: 10 %
complexite=11986672752.0
complexite=16779424752.0
complexite=21572176752.0
complexite=26364928752.0
complexite=31157680752.0
complexite=35950432752.0
complexite=40743184752.0
complexite=45535936752.0
complexite=50328688752.0
complexite=55121440752.0
complexite=59914192752.0
complexite=64706944752.0
complexite=69499696752.0
evaluation after epoch 1
Accuracy of the network on the 10000 test images: 41 %
complexite=83877952752.0
complexite=88670704752.0
complexite=93463456752.0
complexite=98256208752.0
complexite=103048960752.0
complexite=107841712752.0
complexite=112634464752.0
complexite=117427216752.0
complexite=122219968752.0
complexite=127012720752.0
complexite=131805472752.0
complexite=136598224752.0
complexite=141390976752.0
evaluation after epoch 2
Accuracy of the network on the 10000 test images: 51 %
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:    cat   car   car  ship
complexiteImage=11986672752.0
complexiteEpoque=155769232752.0
complexiteTotale=299551792752.0
Accuracy of the network on the 10000 test images: 51 %
Accuracy of plane : 39 %
Accuracy of   car : 74 %
Accuracy of  bird : 21 %
Accuracy of   cat : 39 %
Accuracy of  deer : 28 %
Accuracy of   dog : 33 %
Accuracy of  frog : 79 %
Accuracy of horse : 62 %
Accuracy of  ship : 68 %
Accuracy of truck : 71 %


Resultat 2:
Files already downloaded and verified
Files already downloaded and verified
cuda:0
truck   dog  deer   car
Accuracy of the network on the 10000 test images: 10 %
complexite=11986672752.0
complexite=16779424752.0
complexite=21572176752.0
complexite=26364928752.0
complexite=31157680752.0
complexite=35950432752.0
complexite=40743184752.0
complexite=45535936752.0
complexite=50328688752.0
complexite=55121440752.0
complexite=59914192752.0
complexite=64706944752.0
complexite=69499696752.0
evaluation after epoch 1
Accuracy of the network on the 10000 test images: 46 %
complexite=83877952752.0
complexite=88670704752.0
complexite=93463456752.0
complexite=98256208752.0
complexite=103048960752.0
complexite=107841712752.0
complexite=112634464752.0
complexite=117427216752.0
complexite=122219968752.0
complexite=127012720752.0
complexite=131805472752.0
complexite=136598224752.0
complexite=141390976752.0
evaluation after epoch 2
Accuracy of the network on the 10000 test images: 54 %
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:    cat   car  ship  ship
complexiteImage=11986672752.0
complexiteEpoque=155769232752.0
complexiteTotale=299551792752.0
Accuracy of the network on the 10000 test images: 54 %
Accuracy of plane : 59 %
Accuracy of   car : 69 %
Accuracy of  bird : 40 %
Accuracy of   cat : 27 %
Accuracy of  deer : 33 %
Accuracy of   dog : 66 %
Accuracy of  frog : 63 %
Accuracy of horse : 52 %
Accuracy of  ship : 71 %
Accuracy of truck : 61 %


Resultat 6 epoch

Files already downloaded and verified
Files already downloaded and verified
cuda:0
horse  deer  ship  bird
Accuracy of the network on the 10000 test images: 10 %
complexite=11986672752.0
complexite=16779424752.0
complexite=21572176752.0
complexite=26364928752.0
complexite=31157680752.0
complexite=35950432752.0
complexite=40743184752.0
complexite=45535936752.0
complexite=50328688752.0
complexite=55121440752.0
complexite=59914192752.0
complexite=64706944752.0
complexite=69499696752.0
evaluation after epoch 1
Accuracy of the network on the 10000 test images: 43 %
complexite=83877952752.0
complexite=88670704752.0
complexite=93463456752.0
complexite=98256208752.0
complexite=103048960752.0
complexite=107841712752.0
complexite=112634464752.0
complexite=117427216752.0
complexite=122219968752.0
complexite=127012720752.0
complexite=131805472752.0
complexite=136598224752.0
complexite=141390976752.0
evaluation after epoch 2
Accuracy of the network on the 10000 test images: 55 %
complexite=155769232752.0
complexite=160561984752.0
complexite=165354736752.0
complexite=170147488752.0
complexite=174940240752.0
complexite=179732992752.0
complexite=184525744752.0
complexite=189318496752.0
complexite=194111248752.0
complexite=198904000752.0
complexite=203696752752.0
complexite=208489504752.0
complexite=213282256752.0
evaluation after epoch 3
Accuracy of the network on the 10000 test images: 57 %
complexite=227660512752.0
complexite=232453264752.0
complexite=237246016752.0
complexite=242038768752.0
complexite=246831520752.0
complexite=251624272752.0
complexite=256417024752.0
complexite=261209776752.0
complexite=266002528752.0
complexite=270795280752.0
complexite=275588032752.0
complexite=280380784752.0
complexite=285173536752.0
evaluation after epoch 4
Accuracy of the network on the 10000 test images: 61 %
complexite=299551792752.0
complexite=304344544752.0
complexite=309137296752.0
complexite=313930048752.0
complexite=318722800752.0
complexite=323515552752.0
complexite=328308304752.0
complexite=333101056752.0
complexite=337893808752.0
complexite=342686560752.0
complexite=347479312752.0
complexite=352272064752.0
complexite=357064816752.0
evaluation after epoch 5
Accuracy of the network on the 10000 test images: 63 %
complexite=371443072752.0
complexite=376235824752.0
complexite=381028576752.0
complexite=385821328752.0
complexite=390614080752.0
complexite=395406832752.0
complexite=400199584752.0
complexite=404992336752.0
complexite=409785088752.0
complexite=414577840752.0
complexite=419370592752.0
complexite=424163344752.0
complexite=428956096752.0
evaluation after epoch 6
Accuracy of the network on the 10000 test images: 63 %
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:    dog   car  ship  ship
complexiteImage=11986672752.0
complexiteEpoque=155769232752.0
complexiteTotale=874682032752.0
Accuracy of the network on the 10000 test images: 63 %
Accuracy of plane : 71 %
Accuracy of   car : 85 %
Accuracy of  bird : 61 %
Accuracy of   cat : 27 %
Accuracy of  deer : 58 %
Accuracy of   dog : 48 %
Accuracy of  frog : 81 %
Accuracy of horse : 68 %
Accuracy of  ship : 71 %
Accuracy of truck : 63 %



resultat 6 epoch et time

Files already downloaded and verified
Files already downloaded and verified
cuda:0
horse   cat   cat horse
Accuracy of the network on the 10000 test images: 10 %
complexite=11986672752.0
complexite=16779424752.0
complexite=21572176752.0
complexite=26364928752.0
complexite=31157680752.0
complexite=35950432752.0
complexite=40743184752.0
complexite=45535936752.0
complexite=50328688752.0
complexite=55121440752.0
complexite=59914192752.0
complexite=64706944752.0
complexite=69499696752.0
evaluation after epoch 1
Accuracy of the network on the 10000 test images: 46 %
complexite=83877952752.0
complexite=88670704752.0
complexite=93463456752.0
complexite=98256208752.0
complexite=103048960752.0
complexite=107841712752.0
complexite=112634464752.0
complexite=117427216752.0
complexite=122219968752.0
complexite=127012720752.0
complexite=131805472752.0
complexite=136598224752.0
complexite=141390976752.0
evaluation after epoch 2
Accuracy of the network on the 10000 test images: 52 %
complexite=155769232752.0
complexite=160561984752.0
complexite=165354736752.0
complexite=170147488752.0
complexite=174940240752.0
complexite=179732992752.0
complexite=184525744752.0
complexite=189318496752.0
complexite=194111248752.0
complexite=198904000752.0
complexite=203696752752.0
complexite=208489504752.0
complexite=213282256752.0
evaluation after epoch 3
Accuracy of the network on the 10000 test images: 57 %
complexite=227660512752.0
complexite=232453264752.0
complexite=237246016752.0
complexite=242038768752.0
complexite=246831520752.0
complexite=251624272752.0
complexite=256417024752.0
complexite=261209776752.0
complexite=266002528752.0
complexite=270795280752.0
complexite=275588032752.0
complexite=280380784752.0
complexite=285173536752.0
evaluation after epoch 4
Accuracy of the network on the 10000 test images: 60 %
complexite=299551792752.0
complexite=304344544752.0
complexite=309137296752.0
complexite=313930048752.0
complexite=318722800752.0
complexite=323515552752.0
complexite=328308304752.0
complexite=333101056752.0
complexite=337893808752.0
complexite=342686560752.0
complexite=347479312752.0
complexite=352272064752.0
complexite=357064816752.0
evaluation after epoch 5
Accuracy of the network on the 10000 test images: 61 %
complexite=371443072752.0
complexite=376235824752.0
complexite=381028576752.0
complexite=385821328752.0
complexite=390614080752.0
complexite=395406832752.0
complexite=400199584752.0
complexite=404992336752.0
complexite=409785088752.0
complexite=414577840752.0
complexite=419370592752.0
complexite=424163344752.0
complexite=428956096752.0
evaluation after epoch 6
Accuracy of the network on the 10000 test images: 63 %
Temps d execution : 269.3908734321594 secondes ---
269.3909499645233
3202390961.0805836
Finished Training
GroundTruth:    cat  ship  ship plane
Predicted:   ship  ship  ship plane
complexiteImage=11986672752.0
complexiteEpoque=155769232752.0
complexiteTotale=874682032752.0
Accuracy of the network on the 10000 test images: 63 %
Accuracy of plane : 62 %
Accuracy of   car : 78 %
Accuracy of  bird : 48 %
Accuracy of   cat : 33 %
Accuracy of  deer : 58 %
Accuracy of   dog : 48 %
Accuracy of  frog : 77 %
Accuracy of horse : 67 %
Accuracy of  ship : 86 %
Accuracy of truck : 68 %


"""