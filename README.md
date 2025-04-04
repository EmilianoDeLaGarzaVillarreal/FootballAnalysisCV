Computer vision - Football analysis:

1 – I found out that my personal machine cannot run the chosen model under normal circumstances, so I modified the Batch Size which is a CRITICAL aspect when training a model. By specifying a batch size of 60% of my VRAM I was able to run the model.
1.1 - Batch size defines the speed of training and the resources used for training.
1.1.1 - Larger batch size leads to faster training and more resources used.
1.1.2 - Smaller batch size lead to slower training but fewer resources used.
1.2 - An advantage of smaller batch size is that it helps prevent overfitting by giving a regularization effect to the model.
(Now to more technical learning) ->
2 - Video utilities – Reading, Saving, and Writing
2.1 - The first step on the project involved reading the the video input using cv2, this is simple enough as I only used the method VideoCapture() and iterated over the result appending the frames to a list.
2.2 – Later in the project I had to save the video and for this I used the fourcc function using the ‘XVID’ format. The fourcc function is used to define the codec, which is how the video frames are compressed and stored.
3 – Tracking Moving Objects – Assigning IDs and Unique Annotations
3.1 - Tracking work by the model predicting the next move of the object in the video, that is why models like ByteTrack have parameters like ‘thresholds’. Also, the model tries to update the object tracks by matching new detection based on spatial proximity and confidence.
3.1.1 In my project I used ByteTrack from the Supervision library, from Roboflow.
3.2 - To use the tracker I first used the normal model I had obtained with the training data before. After calling predict for the frames in the video I added the detections to a list which would be used for the tracker to assign IDs for the objects detected in the video.
3.3 - What is the tracks dictionary? Well, for each frame we create a key – tracks[‘player’][frame_num] - we then can loop through the IDs we got from the tracker and assign a bbox for each ID – tracks[‘player’][frame_num][player_id] = {‘bbox’ = [1,1,1,1]} – This would create and entry for the player id and assign a unique bbox to it. If there is no appearance of an object during a frame then the track for that frame would be empty.
3.4 - Some extra interesting things about the tracker class implemented in the project is that it has a lot of drawing functions, from the ellipses that substituted the rectangle bboxes to the speed and distance traveled annotations.
4 – Estimating Camera Movement – Getting Dynamic Section of Video to Better Estimate Object Movements
4.1 - How does Computer Vision and Camera Movement relate? Well think in the following scenario, if you are recording the 100 meter dash competition from the stand and you are using a camera you will likely need to move the camera to follow the movement of the sprinters. If you do not take into account the movement of your camera and try to get the speed of the sprinters you will end up getting speeds lower than what should be in reality. This is because the objects detected in the frame are moving alongside the frame so a model out of the box would think that the objects are not moving.
4.2 - I used several features in order to obtain an effective estimator of camera movement, I learned about how we sometimes need to mask specific areas of the frame in order to make the model focus into areas of more interest.
4.3 - The way I detected movement was by using the previous frames features, ‘old_features’, and the next frames features, ‘new_features’ and comparing changes in distance. This require a couple of for loops in order to get the individual points of each feature but after obtaining the distance difference old frames and new frames and returning a list with the camera movement for each frame.
4.4 - Then I had to add the objects adjusted positions based on the camera movements, to do this I subtracted the camera movement in the x and y axis from the position of an object at the current frame, this was saved in a new entry called ‘position_adjusted’.
4.5 - Estimating camera movement is a very complex subjects and getting it right requires a lot of parameter optimization, I would like to learn more about more methods of doing this.
5 – Transforming Eagle View to Top-Down View – Reducing Distortion / Standardizing Spatial Object Relationships
5.1 – The input video shows a trapezoidal field of view, for computer vision models oftentimes is better to use a standard top-down view as it is more interpretable and reduces distortion.
5.1.1 – We have the pixel vertices which are assigned from the four corners of the desired view area on the trapezoidal shape from the video. We then have the target vertices, which defines how the field should look. We then get the homography mapping transformation matrix is then calculated by cv2.getPerspectiveTransform.
5.2 – All that is left after applying the perspective transform to the object position is to save it in a new entry, ‘position transformed’.
5.2.1 – Take into account that this is done from the position adjusted from the camera movement.
6 – Object Interpolation – Predicting Small Object Movements
6.1 – In this function we are trying to fill missing values from our ‘ball’ tracks,
we use Pandas interpolate and bfill and make the result the new ‘ball’ tracks.
7 – Speed and Cumulative Distance Estimator – Giving Statistics to Objects
7.1 By first defining a frame window I can now calculate distance between the starting position in a frame[i] and the end position in the last frame, frame [i+frame window]. This will return the distance covered in 5 frames.
7.2 Calculating speed makes use of similar ideas and simply adds the frame rate to the equation, speed = distance/time elapsed.
7.3 We then add the speed and distance as new entries for each object in the tracks dictionary.
8 – Team Assigner – Differentiating Objects Based on Color / Grouping Similar Objects
8.1 I used K-means in order to obtain a cluster of two colors from a cropped image of bbox. In the image the dominant color is the background while the least common is the shirt color of the player.
8.1.1 – To be sure I am selecting the shirt color for the players I got the corners of the bbox which are, in most cases, part of the background. So by defining the background of the image we can get the shirt color as there are only two colors in the clustered image.
8.2 We then get the player color of all the players in a frame and we use a K-means model in those colors in order to determine the two different team colors.
8.3 We can then update the player dictionary with the team id and we won’t need to calculate it again unless the bbox of the player loses its tracking.
8.4 After that we can update the ‘player’ tracks with a new entry for the team the player belongs to and the team color for that team.
9 – Team Ball Assigner – Understanding Object Proximity
9.1 First we measure the center of the bbox for the ball, and in order to get a more precise measurement, we calculate the distance from both bottom ends of a player bbox and get the minimum between those distances. If the distance is less than the maximum distance to be referred to as ‘possessing the ball’ then we say that the player has control of the ball.
9.2 In order to calculate ball possession percentage I created a list where I would append the current team that had the ball, then we calculated the frames that team 1 had control of the ball vs team 2 and divided the result over total number of frames that the ball was in possession of any of the two teams.
10 – The project includes several methods for drawing annotations and decorations to make the video look better. Please refer to the demo in order to see some of these annotation, and if you want to see the code you can check my GitHub.
https://github.com/EmilianoDeLaGarzaVillarreal/FootballAnalysisCV
