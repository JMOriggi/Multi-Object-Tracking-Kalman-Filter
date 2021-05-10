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

## Multi-object tracking


## Applications
- Bats tracking:

![Alt text](/git-docs/bats_app.JPG ) 

- Cells tracking

![Alt text](/git-docs/cells_app.JPG ) 
