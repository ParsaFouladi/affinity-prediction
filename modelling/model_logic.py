import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_shape):
        super(CNNModel, self).__init__()
        
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

# Example usage
input_shape = (3, 400, 400)  # Change this shape as needed
model = CNNModel(input_shape)

criterion = nn.BCEWithLogitsLoss()  # Use nn.CrossEntropyLoss() for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Print the model architecture
print(model)
