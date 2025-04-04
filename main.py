import os

import cv2
import numpy as np

from camera_movement_estimator import CameraMovementEstimator
from player_ball_assigner import PlayerBallAssigner
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from team_assigner import TeamAssigner
from trackers import Tracker
from utils import read_video, save_video
from view_transformer import ViewTransformer


def main():
    # Read video
    video_frames = read_video("input/08fd33_4.mp4")

    tracker = Tracker("training/runsResults/detect/train/weights/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pk1"
    )

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Camera Movement Estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path="stubs/camera_movement_stub.pkl"
    )

    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame
    )

    # View transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate ball positions

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

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

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

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

    # Get ball acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["player"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["player"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["player"][frame_num][assigned_player]["team"]
            )
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw output
    # Draw object tracks
    output_video_frames = tracker.draw__annotations(
        video_frames, tracks, team_ball_control
    )

    # Draw Camera Movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )

    # Draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
