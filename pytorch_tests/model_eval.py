import torch
import torch.nn as nn


class SimpleDropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.dropout(self.fc(x))
        return x


model = SimpleDropoutNet()
model.eval()  # Comment this out to test behavior during training mode

input = torch.randn(1, 10)

# Run multiple times
for _ in range(3):
    print(model(input))
