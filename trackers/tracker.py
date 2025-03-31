import os
import pickle
import sys

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

sys.path.append("../")
from utils import get_bbox_width, get_center_of_bbox


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    # This method returns the bounding boxes for all objects detected in the video
    # and other characteristics such as confidence, and we can add a tracker id
    # by using a tracker such as ByteTrack from Supervision from Botflow
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        lRange = int(len(frames) / batch_size)
        for i in range(0, lRange):
            detections_batch = self.model.predict(
                frames[i * batch_size : (i + 1) * batch_size], conf=0.1
            )
            detections += detections_batch
        return detections

    # In here we first convert our input video with the detected bounding boxes to a format that
    # our tracker from sv can understand. Then we update our detections with the tracker
    # and save them in a dictionary for each object detected in the video.
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            # If path exists we load them with picke and return the tracks
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(
            frames
        )  # Calls for detect_frames method and gets object detections
        tracks = {
            "player": [],
            "referee": [],
            "ball": [],
        }  # This is the creation of the tracking bounding box for each object

        # In here we want to change the frames from simply detecting to actual tracking
        for frame_num, detection in enumerate(detections):
            cls_names = (
                detection.names
            )  # This is the classes names ex: 0:"Player", 1:"Goalkeeper", ...
            cls_names_inv = {
                v: k for k, v in cls_names.items()
            }  # This makes it easier to get class_id from the detection output

            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # track objects
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["player"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = (
                    frame_detection[0].tolist()
                )  # This were obtained from the normal frame detection method and
                cls_id = frame_detection[
                    3
                ]  # They are calling for an item in the list ex. track_id
                track_id = frame_detection[4]

                if (
                    cls_id == cls_names_inv["player"]
                ):  # Here we add the bounding box belonging to each object, now with a tracking id
                    tracks["player"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv["referee"]:
                    tracks["referee"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if (
                    cls_id == cls_names_inv["ball"]
                ):  # We do not worry about ball because only one ball
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:  # Saves tranker output to specified path
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])  # bbox[3] gets the bottom of the bounding box
        x_center, _ = get_center_of_bbox(bbox)  # X_center of the bounding box
        width = get_bbox_width(bbox)  # width

        # We are going to draw and ellipse with cv2 method, ellipse. Two axes necessary, and center
        # is the bottom of the bbox and the center of the bounding box
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,  # Start drawing circle at 45 degrees
            endAngle=235,  # Stop drawing the circle
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                str(track_id),
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        (x, _) = get_center_of_bbox(bbox)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ]
        )

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)

        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    # Here we are gonna change the bounding boxes, and dialogs to a more eye pleasing format
    def draw__annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            if frame_num >= len(tracks["player"]):
                break
            frame = frame.copy()

            player_dict = tracks["player"][frame_num]
            referee_dict = tracks["referee"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw Player
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            output_video_frames.append(frame)
        return output_video_frames
