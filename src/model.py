import torch.nn as nn

class EcoSortMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*96*96,432),
            nn.ReLU(),
            nn.Linear(432,6)
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def build_model(num_classes=6):
    return EcoSortMLP(num_classes = num_classes)