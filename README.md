Important aspects of the project that I can talk about:
1 - Object detection to differentiate players and ball.
1.1 - What is Object detection? - Neural Network able to draw a bounding box around the object - and it is able to say what that object is. - It basically knows where it is inside of the image
1.2 - Current state of the art library for Object Detection: - YOLO - We can use YOLO by using ultralytics.
1.3 - Bounding Boxes. - Can befined by x,y for the area of the image, and w,h for - the width and height of the bounding box. - Alternatively we can use x,y then x,y whith the second pair - of coordinates being used for the bounding box.

2 - Detect the ball constantly, the base model has problems
following the ball and the detection constantly blinks.
2.1 - Constant detection is important if we want to analyze ball - adquisition, and different stats.

3 - Model is detecting people outside the field, we need to fix that.
3.1 - Also differentiate between teams
3.2 - Differentiate referees from player too.

4 - How to read the file formats given by the YOLO model, for example:
4.1 - The ID for the object
4.2 - The center position given by x,y
4.3 - The size of the Bounding Box given by relative size - from the center

5 - Batch Size - CRITICAL aspect when training a model, in this case
I could not train the model in the normal way, but specifying
a batch size of 60% of my VRAM allowed me to run the model even
if it exceeded my computer capabilities.
5.1 - Batch size basically define the speed of training and
the resources used for training.
5.1.1 - Larger batch size leads to faster training and more
resources used.
5.1.2 - Smaller batch size lead to slower training but less
resources used.
5.2 - An advantage of smaller batch size is that it helps prevent
overfitting by giving a regularzation effect to the model.

6 - Epochs is making complete loop around the entire dataset
6.1 - An epoch is made of several iterations, one for each
batch

7 - opencv2 will be used to read images from the video, the videos
we have are 24 frames per second so we are reading 24 images
per second

8 - **init**.py inside a directory will make that directory a package.
This allows us to import every function from any file inside that
directory

9 - Tracking How do we define if a bounding box from a previous frame
is the same bounding box from a previous frame? (same user)
9.1 - With only x,y,x,y coordinates we cannot define if
a bounding box belonged to the same player.
9.2 - We can think of it as moving the tracking box with the
player, instead of creating new bounding boxes
for each frame in a video.
9.3 - A small tracker works on probability of a user being the
the same user based on old_position and new_position

10 - Video utilities
10.1 -read_video accepts a video path and will use cv2 VideoCapture
to make the video in the video into a readable format
then we read the capture which gives out ret(end?) and a
frame which we append to a list of frames (the video)
10.2 - save_video accepts an output_video_frame, which comes
from the tracker package, after we draw the annotations
it transforms the video into fourcc format by using
cv2.VideoWriter, we need to specify how many frames
per second. Then we modify the frames in a for loop
and after for loop finished we can release the video

11 - Write about color clustering... used for assigning players
to team
