import torch
import torch.nn as nn

# Example model definition
class MyAestheticModel(nn.Module):
    def __init__(self):
        super(MyAestheticModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(401408, 1)  # Adjust size depending on input
        )

    def forward(self, x):
        return self.backbone(x)

# Instantiate and save model
model = MyAestheticModel()
torch.save(model.state_dict(), "model.pth")
print("Dummy model.pth saved.")
