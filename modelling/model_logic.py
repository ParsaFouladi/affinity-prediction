import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModelBasic(nn.Module):
    def __init__(self, input_shape):
        super(CNNModelBasic, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second Convolutional Block (repeated 3 times)
        self.conv2 = self._make_layer(64, 128, 3)
        
        # Third Convolutional Block (repeated 2 times)
        self.conv3 = self._make_layer(128, 256, 2)
        
        # Dummy forward pass to calculate the output size after conv layers
        self._initialize_fc(input_shape)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _initialize_fc(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.forward_conv(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)

    def forward_conv(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def forward(self, x):
        # Pass through convolutional layers
        x = self.forward_conv(x)
        
        # Flatten the output
        x = x.view(-1, self.flattened_size)
        
        # Fully Connected Layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# # Example usage
# input_shape = (3, 400, 400)  # Change this shape as needed
# model = CNNModelBasic(input_shape)

# criterion = nn.BCEWithLogitsLoss()  # Use nn.CrossEntropyLoss() for multi-class classification
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Print the model architecture
# print(model)

class DeeperCNNModel(nn.Module):
    def __init__(self, input_shape):
        super(DeeperCNNModel, self).__init__()

        # First Convolutional Block
        self.conv1 = self._make_layer(3, 64, 2)
        
        # Second Convolutional Block
        self.conv2 = self._make_layer(64, 128, 2)
        
        # Third Convolutional Block
        self.conv3 = self._make_layer(128, 256, 3)
        
        # Fourth Convolutional Block
        self.conv4 = self._make_layer(256, 512, 3)
        
        # Fifth Convolutional Block
        self.conv5 = self._make_layer(512, 512, 3)
        
        # Sixth Convolutional Block
        self.conv6 = self._make_layer(512, 1024, 2)

        # Calculate the flattened size for the fully connected layer
        self._initialize_fc(input_shape)

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn_fc3 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 1)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout2d(0.5))  # Adding dropout to convolutional layers
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _initialize_fc(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.forward_conv(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)

    def forward_conv(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

    def forward(self, x):
        # Pass through convolutional layers
        x = self.forward_conv(x)
        
        # Flatten the output
        x = x.view(-1, self.flattened_size)
        
        # Fully Connected Layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)  # No activation function here for regression output
        
        return x
