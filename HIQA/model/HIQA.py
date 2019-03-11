import torch
import torch.nn as nn
from .ResUnit import ResUnit

class HourGlass(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass


class QualityAwareGenerativeNetwork(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class QualityRegressionNetwork(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            3
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            3
        )


    def forward(self, img1, img2, encoder_output):

        
        pass

class HallucinationIQA(nn.Module):
    def __init__(self):
        self.quality_aware_generative_network = \
            QualityAwareGenerativeNetwork()

        self.quality_regression_network = \
            QualityRegressionNetwork()

    def forward(self, image):
        reference_gen, encoder_output = \
            self.quality_aware_generative_network(image)
        
        discrepancy = torch.abs(reference_gen - image)
        score = self.quality_regression_network(
            image, 
            discrepancy, 
            encoder_output
        )

        return score




