import torch
from torch import nn

labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
               'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
               'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}
labels_list = [(value, key) for key, value in labels_dict.items()]
labels_list.sort(key=lambda i: i[0])

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1

        # b = 64 # batchsize
        # Input shape= (b,1,200,200)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape= (b,12,200,200)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape= (b,12,200,200)
        self.relu1 = nn.ReLU()
        # Shape= (b,12,200,200)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the image size be factor 2
        # Shape= (b,12,100,100)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape= (b,20,100,100)
        self.relu2 = nn.ReLU()
        # Shape= (b,20,100,100)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (b,32,100,100)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape= (b,32,100,100)
        self.relu3 = nn.ReLU()
        # Shape= (b,32,100,100)

        self.fc = nn.Linear(in_features=100 * 100 * 32, out_features=num_classes)

        # Feed forward function

    def forward(self, _input):
        output = self.conv1(_input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        # Above output will be in matrix form, with shape (b,32,100,100)

        output = output.view(-1, 32 * 100 * 100)

        output = self.fc(output)

        return output


def review(_input:torch.Tensor, size:int) -> torch.Tensor:
    _input = torch.from_numpy(_input)
    return _input.view(1, 1, size, size).float()


def get_model() -> ConvNet:
    """"""
    # model = ConvNet(num_classes=len(labels_dict)).to(device)
    # model.load_state_dict(torch.load("model_weights.model", map_location=torch.device('cpu')))
    model = torch.load("model1.model", map_location=torch.device('cpu'))
    model.eval()
    return model


def get_output_value(output, last=False) -> str:
    """"""
    # last_letter only used for testing purposes
    m = torch.max(output.data,1)
    key = m.indices.data
    value = labels_list[key][1]
    # if last and last != value:
    #     print(m, output, key, value)
    return value
