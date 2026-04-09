from typing import Tuple, List, Dict
from torch import nn, Tensor
from torchinfo import summary
import torch

class TinyVGG(nn.Module):
    def __init__(self, classes_num: int) -> None:
        super(TinyVGG, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(256*7*7, classes_num)

    def _forward_impl(self, x: Tensor) -> Tensor:  # [32,3,224,224]
        x = self.block1(x)  # [32,64,56,56]
        x = self.block2(x)  # [32,128,14,14]
        x = self.block3(x)  # [32,256,7,7]
        x = torch.flatten(x, start_dim=1)  # [32,256*7*7]
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def forward(self, x:Tensor) -> Tensor:
        return self._forward_impl(x)
    

# net = TinyVGG(classes_num=2)
# summary(net, input_size=(32,3,224,224), col_names=["input_size", "output_size", "num_params", "trainable"], col_width=20)
# torch.save(net.state_dict(), "tinyvgg.pth")
