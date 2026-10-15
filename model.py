### Configuración de la red ResNet-18###

import torch.nn as nn
from torchvision import models

class BreastCancerResNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights="DEFAULT")

        # Cambiar la primera capa para aceptar imágenes en escala de grises en lugar de RGB que es lo que espera
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        #Cambiar la última capa para que en vez de tener 1000 categorías que es como está entrenada ResNet, tenga 2 (maligno/benigno)
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

    def forward(self, x):
        return self.model(x)