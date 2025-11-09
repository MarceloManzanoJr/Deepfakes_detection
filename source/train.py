import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import VideoDataset
from model import CNN_LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = VideoDataset(r"C:\Users\USER\Desktop\Deepfakedataset\testing_SDFVD")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = CNN_LSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
criterion = nn.BCELoss()

print("Training started...")

for epoch in range(5):
    total_loss = 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)

        outputs = model(clips).squeeze()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), r"C:\Users\USER\Desktop\Python_Deepfake clip only\models\test_model.pth")
print("Training complete — model saved!")

