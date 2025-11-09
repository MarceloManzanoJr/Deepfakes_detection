import cv2
import os

def extract_frames_and_create_clips(video_path, output_dir, frames_per_clip=16):
    """
    Extract frames from a video and create folder-based clips for deepfake model training.
    Each clip folder will contain `frames_per_clip` numbered frames.
    """

    # Create output directories
    frames_dir = os.path.join(output_dir, "extracted_frames")
    clips_dir = os.path.join(output_dir, "video_clips")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(clips_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"🎥 Video Loaded")
    print(f"Frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")

    frame_count = 0
    clip_count = 0
    frames_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # save raw frame
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frames_buffer.append(frame)

        # if enough frames, save as a clip folder
        if len(frames_buffer) == frames_per_clip:
            save_clip_frames(frames_buffer, clips_dir, clip_count)
            print(f" Created clip {clip_count} from frames {frame_count-frames_per_clip+1} to {frame_count}")

            frames_buffer = []
            clip_count += 1

        frame_count += 1

        if frame_count % 100 == 0:
            print(f"⏳ Processed {frame_count}/{total_frames} frames")

    # If leftover frames exist, save final clip
    if frames_buffer:
        save_clip_frames(frames_buffer, clips_dir, clip_count)
        print(f"Created final clip {clip_count} with {len(frames_buffer)} frames")

    cap.release()
    cv2.destroyAllWindows()

    print("\nExtraction Complete!")
    print(f"Total frames: {frame_count}")
    print(f"Total clips: {clip_count + 1}")
    print(f"Frames saved at: {frames_dir}")
    print(f"Clip folders saved at: {clips_dir}")


def save_clip_frames(frames, clips_dir, clip_number):
    """Save a list of frames as a clip folder."""
    clip_folder = os.path.join(clips_dir, f"clip_{clip_number:04d}")
    os.makedirs(clip_folder, exist_ok=True)

    for i, frame in enumerate(frames):
        frame_filename = os.path.join(clip_folder, f"frame_{i:03d}.jpg")
        cv2.imwrite(frame_filename, frame)

def main():
    video_path = r"C:\Users\USER\Desktop\Python_Deepfake - improve\data\raw_videos\AQMlhNGnDT96IdmUoszZNUWiQqyu-BX3bxmjdECHBOGhaHs726aD8UAfAtP6MQCGs1ebVOqbS-_kkT8omETKvkyJwj9DhqrfNXIPSku7IQ.mp4"     # << CHANGE THIS
    output_directory = r"C:\Users\USER\Desktop\Python_Deepfake - improve\data\raw_videos"     # << CHANGE THIS
    frames_per_clip = 16

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    print("Starting frame extraction...")
    extract_frames_and_create_clips(video_path, output_directory, frames_per_clip)

if __name__ == "__main__":
    main()