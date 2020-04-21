"""
Resumer : 
conv1:3-6-3
conv2:6-8-3
conv3:8-16-5

Critère d'évaluation : CrossEntropyLoss

"""


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
        #Conv 1
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsConvolution(self, 3, 6, 3, totalSize)
        x = self.conv1(x)
        #Relu
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsRelu(self, totalSize)
        x = F.relu(x)
        #Conv2
        xsize = x.size()
        totalSize = xsize[0]*xsize[1]*xsize[2]*xsize[3]
        ajoutPoidsConvolution(self, 6, 8, 3, totalSize)
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
        ajoutPoidsConvolution(self, 8, 16, 5, totalSize)
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




"""