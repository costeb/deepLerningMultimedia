## Compte Rendu Projet Deep Learning

## Évaluation "en continu" du système

# Dans la partie entraînement du réseau CNN, lister les différentes couches et sous-couches. 
A rédiger dans le cr

# Modifier le programme pour faire l'évaluation après chaque époque et aussi avant la première (faire une fonction spécialisée). Supprimer les autres affichages intermédiaires. 
Fait


## Modification du réseau

Configuration de base du réseau: 
def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Modifier le réseau de manière cohérente : commencer par modifier la taille d'une ou plusieurs couche, plus de cartes (filtres ou plans) dans une couche de convolution ou plus de neurones dans une couche complètement connectée. 

Modifications faites: 
-Baisser taille sortie première couche: 3*5 5*16 16*120 120*84 84*10
-Augmenter taille sortie première couche: 3*8 8*16 16*120 120*84 84*10

# Essayer ensuite d'insérer une couche supplémentaire soit de convolution, soit complètement connectée, soit les deux. 
Ajout couche intermédiaire: 3*4 4*6 6*16 16*120 120*84 84*10

# Dans tous les cas, rester sur deux itérations seulement pour limiter le temps d'exécution et comparer les performances des différentes variantes. Dans un ou deux cas, laisser tourner l'entraînement jusqu'à ce que la performance jusqu'à ce que la fonction de coût (running loss) ne décroisse plus ou plus significativement. Comparer la performance finale du réseau du tutoriel et d'une de vos variantes. 

Fait 10 epochs pour "Base" et "Ajout couche"

# Essayer ensuite des variantes de la fonction de coût (loss), de l'optimiseur et/ou de la fonction d'activation. 

Loss:
De base : Cross Entropy
A tester:
- Mean square error loss
- Smooth L1 Loss
- Negative Log-Likelihood Loss
- Margin Ranking Loss

# Calcul du nombre d'opérations flottantes effectuées pour les passes avant (le nombre d'opérations pour les passes arrières, quand il y en a, est quasiment le même) pour une image ; on comptera une opération pour une addition, pour une multiplication ou pour un maximum, même si ces opérations sont de complexités différentes.

Dans forward: relu = une fonction
Nombre d'opérations pour relu(x) = taille de x

conv(x) = fonction
conv(x) = Cconvolution * entrées * sorties
Supposons une convolution stride 1, no padding
Pour 4 tenseurs d'entrée, 5 de sorties, kernel de taille 3 : 
    Complexité = (3*3) *

Pour x tenseurs d'entrée dans tableau X, y tenseurs de sortie dans tableau Y, kernel de taille k:
    nb_operations = ((k*k)* taille de chaque tenseur x) * taille de Y[]

MaxPooling: système de fenêttre appliquznt la fonction max ( https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks )
nn.MaxPool2d(2, 2) : kernel de taille 2, stride de taille 2
Complexité pooling appliqué à un tenseur: (taille du tenseur / taille kernel au carré) * 1 (opération de max)
Si après conv(3,4,5) appliqué 4 fois (les 4 tenseurs de sortie)

Linéaire:
https://stackoverflow.com/questions/54916135/what-is-the-class-definition-of-nn-linear-in-pytorch