import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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


# class VGG16(nn.Module):
#     def __init__(self, input_shape):
#         super(VGG16, self).__init__()
        
#         # Define the VGG16 architecture
#         self.conv1 = self._make_layer(3, 64, 2)
#         self.conv2 = self._make_layer(64, 128, 2)
#         self.conv3 = self._make_layer(128, 256, 3)
#         self.conv4 = self._make_layer(256, 512, 3)
#         self.conv5 = self._make_layer(512, 512, 3)

#         # Calculate the flattened size for the fully connected layer
#         self.flatten_size = self._flatten_size(input_shape)
        
#         # Fully Connected Layers
#         self.fc1 = nn.Linear(self.flatten_size, 4096)
#         self.fc2 = nn.Linear(4096, 4096)
#         self.fc3 = nn.Linear(4096, 1)

#     def _make_layer(self, in_channels, out_channels, num_blocks):
#         layers = []
#         for _ in range(num_blocks):
#             layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
#             layers.append(nn.BatchNorm2d(out_channels))
#             layers.append(nn.ReLU())
#             in_channels = out_channels
#         layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#         return nn.Sequential(*layers)
    
#     def _flatten_size(self, input_shape):
#         block1 =input_shape[1]
#         pool1 =math.ceil((block1-3)/2 +1)
#         #print(pool1)


#         block2=pool1

#         pool2 =math.ceil((block2-3)/2 +1)
#         #print(pool2)



#         block3=pool2
#         pool3 =math.ceil((block3-3)/2 +1)
#         #print(pool3)


#         block4=pool3
#         pool4 =math.ceil((block4-3)/2 +1)
#         #print(pool4)


#         block5=pool4
#         pool5 =math.ceil((block5-3)/2 +1)
#         #print(pool5)


#         #After flatten 
#         flatten_size= pool5 * pool5 * 512

#         return flatten_size

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
        
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)  # No activation function here for regression output

#         return x

class VGG16(nn.Module):
    def __init__(self, input_shape):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        flatten_size = self._flatten_size(input_shape)

        self.fc14 = nn.Linear(flatten_size, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, 1)
    
    def _flatten_size(self, input_shape):
        block1 =input_shape[1]
        pool1 =math.ceil((block1-3)/2 +1)
        #print(pool1)


        block2=pool1

        pool2 =math.ceil((block2-3)/2 +1)
        #print(pool2)



        block3=pool2
        pool3 =math.ceil((block3-3)/2 +1)
        #print(pool3)


        block4=pool3
        pool4 =math.ceil((block4-3)/2 +1)
        #print(pool4)


        block5=pool4
        pool5 =math.ceil((block5-3)/2 +1)
        #print(pool5)


        #After flatten 
        flatten_size= pool5 * pool5 * 512

        return flatten_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.maxpool(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc14(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x


class VGG16_2(nn.Module):
    def __init__(self, input_shape,num_outputs=1):  # Default is 1 output for regression
        super(VGG16_2, self).__init__()
        
        # Define the convolutional layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        flatten_size = self._flatten_size(input_shape)

        # Define the fully connected layers for regression
        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_outputs),  # Output a single value (or more if multivariate regression)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _flatten_size(self, input_shape):
        block1 =input_shape[1]
        pool1 =math.ceil((block1-3)/2 +1)
        #print(pool1)


        block2=pool1

        pool2 =math.ceil((block2-3)/2 +1)
        #print(pool2)



        block3=pool2
        pool3 =math.ceil((block3-3)/2 +1)
        #print(pool3)


        block4=pool3
        pool4 =math.ceil((block4-3)/2 +1)
        #print(pool4)


        block5=pool4
        pool5 =math.ceil((block5-3)/2 +1)
        #print(pool5)


        #After flatten 
        flatten_size= pool5 * pool5 * 512

        return flatten_size

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




