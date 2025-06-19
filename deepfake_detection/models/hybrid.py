import torch
import torchvision as torchvis
import pennylane as qml


class HQNN(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        # Classical model and layers
        self.pretrainedmodel = torchvis.models.resnet18(
            weights=torchvis.models.ResNet18_Weights.DEFAULT
        )
        self.fc = torch.nn.Linear(1000, 417)
        self.fc1 = torch.nn.Linear(417, 64)

        # Quantum device and layers
        self.qnode = qml.QNode(device.circuit, device.device)

        self.qlayer_1 = qml.qnn.TorchLayer(self.qnode, device.weights())
        self.qlayer_2 = qml.qnn.TorchLayer(self.qnode, device.weights())
        self.qlayer_3 = qml.qnn.TorchLayer(self.qnode, device.weights())
        self.qlayer_4 = qml.qnn.TorchLayer(self.qnode, device.weights())
        self.qlayer_5 = qml.qnn.TorchLayer(self.qnode, device.weights())
        self.qlayer_6 = qml.qnn.TorchLayer(self.qnode, device.weights())
        self.qlayer_7 = qml.qnn.TorchLayer(self.qnode, device.weights())
        self.qlayer_8 = qml.qnn.TorchLayer(self.qnode, device.weights())

        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = self.pretrainedmodel(x)

        x = torch.nn.functional.relu(self.fc(x))
        x = torch.nn.functional.relu(self.fc1(x))

        x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8 = torch.split(x, 8, dim=1)

        x_1 = self.qlayer_1(x_1)
        x_2 = self.qlayer_2(x_2)
        x_3 = self.qlayer_3(x_3)
        x_4 = self.qlayer_4(x_4)
        x_5 = self.qlayer_5(x_5)
        x_6 = self.qlayer_6(x_6)
        x_7 = self.qlayer_7(x_7)
        x_8 = self.qlayer_8(x_8)

        x = torch.cat([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8], axis=1)
        x = torch.nn.functional.softmax(self.fc2(x), dim=1)

        return x
