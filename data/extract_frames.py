import os
import glob
import argparse
import datetime
import time
import av
import tqdm


def extract_frames(video_path):
    video = av.open(video_path)
    for frame in video.decode(0):
        yield frame.to_image()


prev_time = time.time()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default="UCF-101", help="Path to UCF-101 dataset"
    )
    opt = parser.parse_args()
    print(opt)

    time_left = 0
    video_paths = glob.glob(os.path.join(opt.dataset_path, "*", "*.avi"))
    for i, video_path in enumerate(video_paths):
        # ✅ FIX: Use os.path.normpath() and os.sep for cross-platform compatibility
        sequence_type, sequence_name = os.path.normpath(video_path).split(os.sep)[-2:]

        sequence_path = os.path.join(
            f"{opt.dataset_path}-frames", sequence_type, sequence_name
        )

        if os.path.exists(sequence_path):
            continue

        os.makedirs(sequence_path, exist_ok=True)

        # Extract frames
        for j, frame in enumerate(
            tqdm.tqdm(
                extract_frames(video_path),
                desc=f"[{i}/{len(video_paths)}] {sequence_name} : ETA {time_left}",
            )
        ):
            frame.save(os.path.join(sequence_path, f"{j}.jpg"))

        # Determine approximate time left
        videos_left = len(video_paths) - (i + 1)
        time_left = datetime.timedelta(seconds=videos_left * (time.time() - prev_time))
        prev_time = time.time()
