"""
Resumer : 
Resaux de neurone de base

Critère d'évaluation : CrossEntropyLoss

"""

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
        #Conv 1
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsConvolution(self, 3, 4, 3, totalSize)
        x = self.conv1(x)
        #Relu
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsRelu(self, totalSize)
        x = F.relu(x)
        #Conv2
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsConvolution(self, 4, 6, 3, totalSize)
        x = self.conv2(x)
        #Relu
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsRelu(self, totalSize)
        x = F.relu(x)
        #Pool
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsMaxPool(self, 2, totalSize)
        x = self.pool(x)
        #Conv3
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsConvolution(self, 6, 16, 5, totalSize)
        x = self.conv3(x)
        #Relu
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsRelu(self, totalSize)
        x = F.relu(x)
        #Pool
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsMaxPool(self, 2, totalSize)
        x = self.pool(x)

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


criterion = nn.CrossEntropyLoss()


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



"""