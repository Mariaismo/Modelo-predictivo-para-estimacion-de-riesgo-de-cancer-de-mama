### Configuración de la red DenseNet-121###

import torch.nn as nn
from torchvision import models


class BreastCancerDenseNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.densenet121(weights="DEFAULT")

        # Cambiar la primera capa para aceptar imágenes en escala de grises en lugar de RGB que es lo que espera
        self.model.features.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        #Cambiar la última capa para que en vez de tener 1000 categorías que es como está entrenada DenseNet, tenga 2 (maligno/benigno)
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features,
            2
        )

    def forward(self, x):
        return self.model(x)