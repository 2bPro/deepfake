import torch
import torchvision as torchvis


class ClassicalModel(torch.nn.Module):
    def __init__(self, offline=None):
        super().__init__()

        if offline:
            self.pretrainedmodel = torchvis.models.resnet18()
            self.pretrainedmodel.load_state_dict(torch.load(offline))
            self.pretrainedmodel.eval()
        else:
            self.pretrainedmodel = torchvis.models.resnet18(
                weights=torchvis.models.ResNet18_Weights.DEFAULT
            )

        self.fc = torch.nn.Linear(1000, 64)
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = self.pretrainedmodel(x)
        x = torch.nn.functional.relu(self.fc(x))
        x = torch.nn.functional.softmax(self.fc2(x), dim=1)

        return x
