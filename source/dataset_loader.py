import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import CNN_LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128,128))
])

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16):
        self.samples = []
        self.num_frames = num_frames

        for label, folder in enumerate(["real", "fake"]):
            class_dir = os.path.join(root_dir, folder)
            for file in os.listdir(class_dir):
                if file.endswith(".mp4"):
                    self.samples.append((os.path.join(class_dir, file), label))

    def _load_clip(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // self.num_frames)

        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frames.append(frame)

        cap.release()

        if len(frames) < self.num_frames:
            # pad last frame if video too short
            while len(frames) < self.num_frames:
                frames.append(frames[-1])

        return torch.stack(frames)  # [16, 3, 128, 128]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        frames = self._load_clip(path)
        return frames, torch.tensor(label, dtype=torch.float32)

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
