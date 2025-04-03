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

8 - \_\_init\_\_.py inside a directory will make that directory a package.
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
the same user based on old\_position and new\_position

10 - Video utilities
10.1 -read\_video accepts a video path and will use cv2 VideoCapture
to make the video in the video into a readable format
then we read the capture which gives out ret(end?) and a
frame which we append to a list of frames (the video)
10.2 - save\_video accepts an output\_video\_frame, which comes
from the tracker package, then we draw the annotations and
it transforms the video into fourcc format by using
cv2.VideoWriter, we need to specify how many frames
per second. Then we modify the frames in a for loop
and after for loop finished we can release the video
10.2.1 - output\_video\_frame.shape gives the x and y dimensions
of the entire video frame.

11 - Tracker
11.1 - in this package we have the model we are going to use and
the default tracker, ByteTrack from supervision.
11.2 First method called from main get\_get\_object\_tracks gets the
frames from the videos and assigns bounding boxes for everyone
in the video, this is done by calling detect\_frames.
11.3 Detect\_frames - Simply calls model.predict to get video
with bboxes.
11.4 After getting the video with the bboxes we create the tracks
for each object in the model.
11.4.1 What are tracks? It is a dictionary with the key being the
object, ex. 'player', where each bbox is assigned a place in the
dictionary, ex. tracks['player'][frame\_num][track\_id] is a new entry
in the dictionary, or the same entry is updated for the next frame,
sharing the tracking id.

12 - Image Transformation
12.1 Cropping an image, we get the first frame of the video, then
we crop the image based on the dimension of the bbox of a random
player, then use cv2.imwrite to save the cropped image.
12.2 K-Mean team assigning, We use the cropped image and get only
the upper half, to only get shirt color as a predominant color.
we the labels of the kmean and reshape them back to the image
shape. A creative way of differentiating background from
the player is getting the four corners of the image, where the
player is least likely to be.

13 - TeamAssigner
13.1 We create a teamAssigner class initialized with a team color
and a player\_team\_directory.
13.2 - assign\_team\_color gets self, the frame and player\_detection,
this last argument refers to the tracked bbox in the frame. We
call get\_player\_color in a loop for each player detected in the 
tracked frame, refer to Image transformation to see how this method works.

11 - Write about color clustering... used for assigning players
to team
