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