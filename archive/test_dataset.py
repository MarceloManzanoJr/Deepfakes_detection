from dataset_loader import DeepfakeFramesDataset
from torch.utils.data import DataLoader

dataset_path = r"C:\Users\USER\Desktop\Deepfakedataset_frames"

dataset = DeepfakeFramesDataset(dataset_path, frames_per_clip=16)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

print(f"Total clips: {len(dataset)}")

for clips, label in loader:
    print("Clip shape:", clips.shape)   # Expected: [1, 16, 3, 128, 128]
    print("Label:", label.item())       # 0=real, 1=fake
    break
