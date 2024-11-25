# Real-Time Camera Pose Estimator

This program provides a solution for estimating the 6-DoF (Degrees of Freedom) camera pose, which includes position and orientation relative to a global coordinate system defined by a chessboard pattern. It operates in **real-time**.

---

## Prerequisites

### Hardware Requirements
- **Camera:** Basler camera.
- **Chessboard Pattern:** A printed or projected chessboard pattern with known dimensions.

### Software Requirements
- **Pylon SDK:** Required for connecting the camera to the computer.  
  Download it here: [Pylon Viewer](https://www.baslerweb.com/en/software/pylon/).

---

## Installation

1. Clone the repository by using git clone with the repository link.
```bash
git clone https://github.com/YunhyeokHan/PoseEstimator.git
```
2. (Optional) Create a Python 3.9 environment by using conda create command with the name `PoseEstimator` and Python version `3.9`.  
```bash
conda create --name PoseEstimator python=3.9
```
3. Install the required Python libraries by using pip install with the `requirements.txt` file.
```bash
pip install -r requirements.txt
```
---

## Usage

### Camera Calibration

Before running the pose estimation, the camera must be calibrated using a chessboard pattern.

1. Start the calibration process by running the camera calibrator script and providing arguments for the checkerboard dimensions, square size, and exposure time. 

- Define the number of corner points in the rows and columns of the pattern (`--checkerboard <rows> <columns>`). 
- Square size: Specify the size of each square on the chessboard in millimeters (`--square_size <size>`).
- Exposure time: Set the camera exposure time in microseconds (`--exposure_time <value>`).
```bash
python ¨PoseEstimator/camera_calibrator.py --checkerboard 7 5 --square_size 10 --exposure_time 20000
```
2. Place the chessboard in the camera's view and verify that all corner points are detected. The program will display real-time feedback.
<p align="center">
  <img src="./doc/build/img/image_0.png" alt="image info" width="1200"/>
</p>

3. Press Spacebar to save detected points for calibration. Move the chessboard to different positions and orientations, capturing sufficient data points.
4. Press ESC to end the capture session. The program will perform intrinsic camera calibration and save the results for future use.

---

### Pose Estimation

Once the camera is calibrated, you can run the pose estimation program.

1. Start the pose estimator by running the pose estimation script with the same arguments as the calibration process.
- Number of corner points in the rows and columns of the pattern (`--checkerboard <rows> <columns>`). 
- Square size: Specify the size of each square on the chessboard in millimeters (`--square_size <size>`).
- Exposure time: Set the camera exposure time in microseconds (`--exposure_time <value>`).
```bash
python PoseEstimator/pose_estimator.py --checkerboard 7 5 --square_size 10 --exposure_time 20000
```
2.The program will display real-time camera feed and calculate the 6-DoF pose relative to the chessboard pattern. Ensure that the pattern is visible to the camera and its corners are detected. Position/orientation of camera is defined with respect to Global XYZ axis of chessboard pattern (XYZ axes are depicted as R G B in the image).
<p align="center">
  <img src="./doc/build/img/pose_estimation.png" alt="image info" width="1200"/>
</p>

<p align="center">
  <img src="./doc/build/img/pose_estimation_plot.png" alt="image info" width="1200"/>
</p>

3. Press ESC to terminate the program.

---

## Notes

- Calibration results are saved automatically and reused for pose estimation.  
- Ensure the chessboard pattern dimensions and exposure settings are correctly provided in all steps for accurate results.  
- For real-time performance, use a compatible camera and a well-lit environment.
