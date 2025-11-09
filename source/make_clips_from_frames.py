import os
import shutil

root = r"C:\Users\USER\Desktop\Deepfakedataset_frames"
frames_per_clip = 16

for label in ["videos_real", "videos_fake"]:
    base = os.path.join(root, label)
    videos = os.listdir(base)

    clip_index = 0
    for vid in videos:
        vid_path = os.path.join(base, vid)
        frames = sorted(os.listdir(vid_path))

        for i in range(0, len(frames), frames_per_clip):
            clip_folder = os.path.join(base, f"{label}_{clip_index:05d}")
            os.makedirs(clip_folder, exist_ok=True)

            for j in range(i, min(i + frames_per_clip, len(frames))):
                src = os.path.join(vid_path, frames[j])
                dst = os.path.join(clip_folder, frames[j])
                shutil.copy(src, dst)

            clip_index += 1

print("Done creating 16-frame clip folders.")
