# UNI: tmn2134
import torch 
import torch.nn as nn 
import torch.nn.functional as F


class AttentionPoolingNet(nn.Module):
    def __init__(self, input_channels, output_size):
        super(AttentionPoolingNet, self).__init__()
        self.input_channels = input_channels
        self.output_size = output_size
        
        self.attention_weights = nn.Linear(input_channels, 1)
        self.fc = nn.Linear(input_channels, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, input_channels, timestep)
        batch_size, input_channels, timestep = x.size()
        
        # Calculate attention weights
        attention_scores = self.attention_weights(x.permute(0, 2, 1))  # (batch_size, timestep, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # Apply softmax along timestep dimension
        
        # Apply attention weights to input
        weighted_input = torch.bmm(x, attention_weights)  # (batch_size, input_channels, 1)
        
        # Flatten and apply fully connected layer
        weighted_input = weighted_input.view(batch_size, input_channels)
        output = self.fc(weighted_input)
        
        return output

class SPKConv(nn.Module):
    def __init__(self, num_classes: int, weight=None):
        super(SPKConv, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(80, 160, kernel_size=7, padding=1),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(160, 256, kernel_size=7, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(256, 512, kernel_size=7, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(512, 512, kernel_size=7, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(4),

            # nn.Conv1d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.MaxPool1d(4),
        )
        # print(self.features(torch.randn(1, 80, 1500)).shape)

        b, c, t = self.features(torch.randn(1, 80, 3000)).shape

        self.attnPool = AttentionPoolingNet(c, c)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, mel):
        # feature extraction 
        x = self.features(mel)

        # attention pooling
        x = self.attnPool(x)
        x = self.dropout(x)

        # classification
        x = self.classifier(x)
        return x
