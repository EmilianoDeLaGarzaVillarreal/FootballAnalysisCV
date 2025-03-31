import os

import cv2

from team_assigner import TeamAssigner
from trackers import Tracker
from utils import read_video, save_video


def main():
    # Read video
    video_frames = read_video("input/08fd33_4.mp4")

    tracker = Tracker("training/runsResults/detect/train/weights/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pk1"
    )

    # save cropped image of a player
    if os.path.exists("output_videos/cropped_image.jpg"):
        print("Cropped Image in existence")
    else:
        for _, player in tracks["player"][0].items():
            bbox = player["bbox"]
            frame = video_frames[0]

            # crop bbox from the frame
            cropped_image = frame[
                int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])
            ]

            # save cropped image
            cv2.imwrite("output_videos/cropped_image.jpg", cropped_image)

            break

    # Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["player"][0])

    for frame_num, player_track in enumerate(tracks["player"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )

            tracks["player"][frame_num][player_id]["team"] = team
            tracks["player"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team]
            )

    # Draw output
    # Draw object tracks
    output_video_frames = tracker.draw__annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
