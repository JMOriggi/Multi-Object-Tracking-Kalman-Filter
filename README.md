# Multi-Object-Tracking-Kalman-Filter

## About Kalman Filter
Kalman filtering, also known as linear quadratic estimation (LQE), is an algorithm 
that uses a series of measurements observed over time, containing statistical 
noise and other inaccuracies, and produces estimates of unknown variables that tend 
to be more accurate than those based on a single measurement alone, by estimating 
a joint probability distribution over the variables for each timeframe.

Essentially the measurement has some noise and with the kalman filter we want ot eliminate the noise.

Kalman provide an estimate of want I cannot measure precisely with what I can measure at the moment.
Also use to fuse multiple noisy measurement to best find the car position for example.
Kalman is a linear model, so good for object that moves in a linear way (like bats). Cells for example moves in a non linear way,
in that case we should consider other methods (complex non linear Kalman filter).

![Alt text](/git-docs/kalman.JPG )

## Multi-object tracking
I used Global Nearest Neighbor Standard Filter (GNNSF) as multi object tracking algorithm. Using a derived cost matrix I'm then able to derive the best match of each track with each measurement. Each track should have a measurement associated. Matrix formulation of the algorithm implemented in the image:

![Alt text](/git-docs/multitrack.JPG )

## Applications
- Bats tracking: using multitracking and kalman filter estimation. Only for this task we considered the segmentation and localization already provided. In the details when a track doesn’t have a clear close position available (we search in the cost matrix the position with minimum cost; cost computed between the expected position decided by the kalman filter and the list of available localization in the frame), we will trust the kalman filter estimate as current position. If the track stays for more than 5 frames in this state we will consider the track lost and we will drop it.

![Alt text](/git-docs/bats_app.JPG ) 

- Cells tracking: The process is pretty much the same as bats and we use multitracking. However, since cells do not leave the frame, we cannot really lose their track in that way but for the occlusion frames, we estimate the position of the cells using a random coordinate generator.

![Alt text](/git-docs/cells_app.JPG ) 
