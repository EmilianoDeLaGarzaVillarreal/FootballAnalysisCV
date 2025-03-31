import cv2


def read_video(video_path):
    cap = cv2.VideoCapture(
        video_path
    )  # Get capture of video to format that cv2 can read it
    frames = []
    ret = True  # Used when video ends becomes false
    while ret:
        ret, frame = (
            cap.read()
        )  # outputs bool for video ending, and frame for images in video
        frames.append(frame)  # We create a list of frames in the video
    return frames  # return the frames in a list


def save_video(output_video_frame, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(
        *"XVID"
    )  # This does exist, it basically writes it to the format fourcc
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,
        (
            output_video_frame[0].shape[1],
            output_video_frame[0].shape[0],
        ),  # output video shape
    )  # This writes the video to an output based on arguments passed
    for frame in output_video_frame:
        out.write(frame)  # We use this function to modify each frame in video
    out.release()  # release the video
