import torch
import torch.nn as nn
import torchvision.models as models

class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1):
        super(CNN_LSTM, self).__init__()

        
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.cnn = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # Freeze CNN layers to train faster on CPU
        for param in self.cnn.parameters():
            param.requires_grad = False

        cnn_output_size = 1280  # MobileNetV2 output channels

        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, frames, C, H, W)
        batch_size, frames, C, H, W = x.size()
        x = x.view(batch_size * frames, C, H, W)

        with torch.no_grad():
            x = self.cnn(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)

        x = x.view(batch_size, frames, -1)
        lstm_out, _ = self.lstm(x)
        final_output = lstm_out[:, -1, :]  # last frame output

        out = self.fc(final_output)
        return torch.sigmoid(out)
