from trackers import Tracker
from utils import read_video, save_video


def main():
    # Read video
    video_frames = read_video("input/08fd33_4.mp4")

    tracker = Tracker("training/runsResults/detect/train/weights/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pk1"
    )

    # Draw output
    # Draw object tracks
    output_video_frames = tracker.draw__annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
