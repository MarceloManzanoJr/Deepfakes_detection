import os
import cv2
import torch
import numpy as np
from model import CNN_LSTM as DeepfakeDetector 
from torchvision import transforms
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128,128))
])

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_frames)


    video_name = os.path.splitext(os.path.basename(video_path))[0]
    temp_dir = f"data/temp_frames/{video_name}"
    os.makedirs(temp_dir, exist_ok=True)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret: 
            break

       
        frame_path = os.path.join(temp_dir, f"frame_{i:04}.jpg")
        cv2.imwrite(frame_path, frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)

    cap.release()
    return torch.stack(frames), temp_dir  # returns frames + folder path

def save_anomaly_frames(frame_folder, video_name):
    output_dir = f"data/results/anomalies/{video_name}"
    os.makedirs(output_dir, exist_ok=True)

    for f in os.listdir(frame_folder):
        src = os.path.join(frame_folder, f)
        dst = os.path.join(output_dir, f)
        shutil.copy(src, dst)

    print(f"[✅] Suspicious frames saved to: {output_dir}")

def predict(video_path):
    model = DeepfakeDetector().to(device)
    model.load_state_dict(torch.load(r"models\deepfake_model_finetuned_20251108_195623.pth", map_location=device))
    model.eval()

    frames, frame_folder = extract_frames(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    clip = frames.unsqueeze(0).to(device)  # [1,16,3,128,128]

    with torch.no_grad():
        output = model(clip)
        prob = torch.sigmoid(output).item()

    print(f"\nVideo: {video_name}")
    print(f"Fake probability: {prob:.4f}")

    if prob > 0.6:
        print("Deepfake Detected — extracting suspicious frames...")
        save_anomaly_frames(frame_folder, video_name)
    else:
        print("Real Video — no anomaly frames saved.")

if __name__ == "__main__":
    test_video = r"C:\Users\USER\Desktop\Python_Deepfake - improve\data\raw_videos\AQMlhNGnDT96IdmUoszZNUWiQqyu-BX3bxmjdECHBOGhaHs726aD8UAfAtP6MQCGs1ebVOqbS-_kkT8omETKvkyJwj9DhqrfNXIPSku7IQ.mp4"
    predict(test_video)
