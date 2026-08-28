"""
model.py

The exact same AudioCNN class from your training notebook.

WHY WE NEED THE CLASS DEFINITION HERE, NOT JUST THE SAVED WEIGHTS:
Remember - torch.save(model.state_dict(), ...) only saved the LEARNED
NUMBERS (weights/biases), not the class/blueprint itself. To use those
numbers again, we first need to build a fresh, empty AudioCNN object
(same architecture, randomly-initialized), and then pour the saved
numbers into it with load_state_dict(). The architecture defined here
must match the training one exactly, or the saved weights won't fit.
"""

import torch.nn as nn
import torch.nn.functional as F


class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * 1 * 1, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
