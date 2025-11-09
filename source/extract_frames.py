import cv2
import os

dataset_path = r"C:\Users\USER\Desktop\Deepfakedataset\SDFVD"  # your dataset
output_path = r"C:\Users\USER\Desktop\Deepfakedataset_frames"  # new folder for frames

os.makedirs(output_path, exist_ok=True)

for class_folder in ["videos_real", "videos_fake"]:
    input_dir = os.path.join(dataset_path, class_folder)
    output_dir = os.path.join(output_path, class_folder)
    os.makedirs(output_dir, exist_ok=True)

    for video_name in os.listdir(input_dir):
        if video_name.endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(input_dir, video_name)

            video_name_no_ext = os.path.splitext(video_name)[0]
            video_output_folder = os.path.join(output_dir, video_name_no_ext)
            os.makedirs(video_output_folder, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            frame_index = 1

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_file = os.path.join(video_output_folder, f"frame_{frame_index:04d}.jpg")
                cv2.imwrite(frame_file, frame)
                frame_index += 1

            cap.release()
            print(f"Frames extracted from {video_name}")

print("All frames extracted successfully!")
