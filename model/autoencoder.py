import torch
import torch.nn as nn
import torchvision 

class AutoEncoder(nn.Module):
    def __init__(self, 
                 input_ch: int, 
                 bottleneck_ch: int,  
                 mode='linear') -> nn.Module:

        super(AutoEncoder, self).__init__()
        self.input_ch = input_ch
        self.output_ch = input_ch
        self.mode = mode
        self.bottleneck_ch = bottleneck_ch

        if self.mode == 'linear':
            self.encoder = nn.Sequential(
                    nn.Linear(self.input_ch, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.bottleneck_ch),
                    nn.ReLU()
                    )
            
            self.decoder = nn.Sequential(
                    nn.Linear(self.bottleneck_ch, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, self.output_ch)
                    )
            
        if self.mode == 'cnn':
            self.encoder = nn.Sequential(
                    nn.Conv2d(self.input_ch, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    )
            
            self.decoder = torch.nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(64, self.output_ch, kernel_size=3, stride=1, padding=1)
                    ) 
            
    def forward(self, x: torch.tensor) -> torch.tensor:
        output = self.encoder(x)
        output = self.decoder(output)
        return output