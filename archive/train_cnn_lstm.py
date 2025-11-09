import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from PIL import Image
from sklearn.model_selection import train_test_split

# ---------------------- SETTINGS -------------------------
DATASET_PATH = r"C:\Users\USER\Desktop\Deepfakedataset"
SEQ_LEN = 16
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE = "deepfake_cnn_lstm.pth"
IMG_SIZE = 224
# ---------------------------------------------------------

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Dataset Loader
class DeepfakeDataset(Dataset):
    def __init__(self, clip_paths, labels):
        self.clip_paths = clip_paths
        self.labels = labels

    def load_clip(self, folder):
        frames = sorted(os.listdir(folder))[:SEQ_LEN]
        imgs = []

        for f in frames:
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            imgs.append(transform(img))

        while len(imgs) < SEQ_LEN:
            imgs.append(imgs[-1])

        return torch.stack(imgs)

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        clip_dir = self.clip_paths[idx]
        label = self.labels[idx]
        clip_tensor = self.load_clip(clip_dir)
        return clip_tensor, torch.tensor(label, dtype=torch.float32)

# Model: CNN + LSTM
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        base_model = resnet18(weights="IMAGENET1K_V1")
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])
        self.lstm = nn.LSTM(512, 256, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.cnn(x).view(B, T, 512)
        _, (h, _) = self.lstm(x)
        x = h[-1]
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x.squeeze()

def load_data():
    clips = []
    labels = []

    for label, folder in [(0, "real"), (1, "fake")]:
        folder_path = os.path.join(DATASET_PATH, folder)
        for c in os.listdir(folder_path):
            full = os.path.join(folder_path, c)
            if os.path.isdir(full):
                clips.append(full)
                labels.append(label)
    return clips, labels

clips, labels = load_data()
train_clips, val_clips, y_train, y_val = train_test_split(clips, labels, test_size=0.2, stratify=labels)

train_ds = DeepfakeDataset(train_clips, y_train)
val_ds = DeepfakeDataset(val_clips, y_val)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = CNN_LSTM().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("🚀 Training Started")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for clips, labels in train_dl:
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)
        preds = model(clips)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_dl):.4f}")

torch.save(model.state_dict(), MODEL_SAVE)
print(f"\n✅ Model saved as {MODEL_SAVE}")
