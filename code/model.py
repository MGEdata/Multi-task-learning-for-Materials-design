import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class VectorToScalarCNN(nn.Module):
    def __init__(self):
        super(VectorToScalarCNN, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Max Pooling Layers
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)


class EncoderModel(nn.Module):


    def __init__(self, params,task_features):
        super(EncoderModel, self).__init__()

        self.preders = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(task_features[task], params.hiddens),
                    nn.ReLU(),
                ) for task in list(task_features.keys())
            }
        )

        self.shared_feature_extractor = nn.Sequential(
            nn.Linear(params.hiddens, int(params.hiddens / 2)),
            nn.ReLU(),
            nn.Linear(int(params.hiddens / 2), int(params.hiddens / 2)),
            nn.ReLU()
        )

    def forward(self, x):
        # print(x.size())
        features_scaler = self.preders(x.float())
        shared_features = self.shared_feature_extractor(features_scaler.float())
        return shared_features



class EncoderModel_cc(nn.Module):
    def __init__(self):
        super(EncoderModel_cc, self).__init__()

        # First CNN Branch
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Second CNN Branch (identical to the first one)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels=24, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )



    def forward(self, x):
        # Apply the first CNN branch
        x = x.float()
        out1 = self.branch1(x)

        # Apply the second CNN branch
        out2 = self.branch2(x)

        # Combine the outputs of both branches element-wise (e.g., addition or concatenation)
        # In this example, we concatenate the outputs along the channel dimension
        combined = torch.cat([out1, out2], dim=1)  # Concatenate along the channel dimension

        # Global average pooling to reduce spatial dimensions
        x = torch.mean(combined, dim=2)  # Calculate the mean along the sequence_length dimension

        return x

# # Create an instance of the model
# model = EncoderModel()
#
# # Example input tensor with shape (n, 12, 1536)
# input_tensor = torch.randn((64, 12, 1536))  # Adjust the batch size (64) as needed
#
# # Forward pass through the model
# output_tensor = model(input_tensor)
#
# # The output_tensor will have shape (n, 64)
# print(output_tensor.shape)






