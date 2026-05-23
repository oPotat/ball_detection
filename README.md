[README_BallGoal.md](https://github.com/user-attachments/files/28178971/README_BallGoal.md)
# Ball-to-Goal Robot Vision

A real-time computer vision system that detects a ball and a goal in a live camera feed, estimates distances to both, and outputs steering commands to guide a robot toward the goal.

## Features

- **Ball detection** — HSV colour segmentation supporting white, green, blue, and red targets
- **Goal detection** — contour filtering by area, aspect ratio, and solidity to isolate a U-shaped or rectangular goal marker
- **Distance estimation** — focal-length formula for both ball and goal using known physical dimensions
- **Trajectory overlay** — dashed line + arrow from ball to goal with a suggested robot approach position
- **Steering output** — computes a screen-space angle and calls `turn_left()`, `turn_right()`, or `forward()`
- **Temporal smoothing** — 8-frame rolling average on goal position to reduce jitter
- **Ball trail** — fading path showing recent ball movement

## Setup

```bash
pip install opencv-python numpy
```

A physical camera (USB or IP stream) is required.

## Configuration

All constants are at the top of `Ball_Goal.py`:

| Parameter | Default | Description |
|---|---|---|
| `STREAM_URL` | `http://192.168.1.8:8080/video` | IP camera URL |
| `TARGET_COLOR` | `"white"` | Ball colour: `white`, `green`, `blue`, or `red` |
| `KNOWN_DIAMETER_CM` | `4.0` | Real diameter of the ball (ping pong = 4 cm) |
| `FOCAL_LENGTH_PX` | `433.0` | Calibrated focal length in pixels |
| `KNOWN_GOAL_WIDTH_CM` | `30.0` | Real width of the goal in cm |
| `MIN_AREA / MAX_AREA` | `5000 / 60000` | Contour area bounds for goal detection |
| `SHOW_MASK` | `True` | Show HSV mask windows for debugging |

## Calibration

To find your camera's focal length, enable calibration mode:

```python
CALIBRATION_MODE  = True
KNOWN_DISTANCE_CM = 20.0   # Hold the ball exactly this far from the camera
```

Run the script and read the printed focal length, then paste it into `FOCAL_LENGTH_PX` and set `CALIBRATION_MODE = False`.

## Camera Source

The script defaults to webcam (`cv2.VideoCapture(0)`). To use an IP camera stream instead, comment out that line and uncomment:

```python
cap = cv2.VideoCapture(STREAM_URL)
```

## Robot Integration

Implement the three stub functions with your robot's motor commands:

```python
def turn_right(): ...
def turn_left():  ...
def forward():    ...
```

The steering threshold is ±10°. Angles beyond that trigger a turn; within it, the robot moves forward.

## Controls

Press `q` to quit.
