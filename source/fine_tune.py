import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import datetime
from model import CNN_LSTM
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VideoDataset(Dataset):
    def __init__(self, video_folder, frames_per_clip=16, transform=None):
        self.video_folder = video_folder
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.videos = []
        self.labels = []

        for label, category in enumerate(["real", "fake"]):
            path = os.path.join(video_folder, category)
            if not os.path.exists(path):
                continue
            for file in os.listdir(path):
                if file.lower().endswith((".mp4", ".avi", ".mov")):
                    self.videos.append(os.path.join(path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]

        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // self.frames_per_clip)

        for i in range(self.frames_per_clip):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if not ret:
                frame = 255 * np.ones((128,128,3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        cap.release()
        clip_tensor = torch.stack(frames)  # [frames, C, H, W]
        return clip_tensor, torch.tensor(label, dtype=torch.float32)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128,128))
])

new_dataset_path = r"C:\Users\USER\Desktop\Deepfakedataset\testing_SDFVD"
pretrained_model_path = r"models\deepfake_model_finetuned_20251108_192151.pth"
frames_per_clip = 16
batch_size = 2
num_epochs = 5
learning_rate = 1e-4

dataset = VideoDataset(new_dataset_path, frames_per_clip=frames_per_clip, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CNN_LSTM().to(device)
model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
model.train()

# unfreeze last CNN layers

# for name, param in model.cnn.named_parameters():
#     if "18" in name or "19" in name:
#         param.requires_grad = True


criterion = nn.BCELoss() 
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0.0
    for clips, labels in dataloader:
        clips = clips.to(device)        # [B, frames, C, H, W]
        labels = labels.to(device).unsqueeze(1)  # [B,1]

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clips.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f}")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fine_tuned_model_path = f"models/deepfake_model_finetuned_{timestamp}.pth"
torch.save(model.state_dict(), fine_tuned_model_path)
print(f"✅ Fine-tuning complete! Model saved to {fine_tuned_model_path}")
