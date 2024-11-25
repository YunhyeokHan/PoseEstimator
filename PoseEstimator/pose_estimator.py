import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pypylon import pylon
import argparse
import matplotlib.ticker as ticker

# ----------------------- Utility Functions -----------------------

import math
import numpy as np

def plotXYZ(img, rvec,tvec,newcameramatrix):
    axis = np.float32([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]).reshape(-1,3)*500

    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, newcameramatrix, None)

    img = cv2.line(img, tuple(imgpts[0].ravel().astype(int)), tuple(imgpts[1].ravel().astype(int)), color=(0,0,255), thickness=3)
    img = cv2.line(img, tuple(imgpts[0].ravel().astype(int)), tuple(imgpts[2].ravel().astype(int)), color=(0,255,0), thickness=3)
    img = cv2.line(img, tuple(imgpts[0].ravel().astype(int)), tuple(imgpts[3].ravel().astype(int)), color=(255,0,0), thickness=3)

    return img

def rotationMatrixToEulerAngles(R):
    """
    Convert a rotation matrix to Euler angles (XYZ convention).

    Args:
        R (numpy.ndarray): A 3x3 rotation matrix.

    Returns:
        numpy.ndarray: Euler angles (in radians) as [roll, pitch, yaw].
    """
    if abs(R[2, 0]) != 1:
        yaw2 = math.asin(R[2, 0])  # Calculat yaw (Y-axis rotation)

        pitch2 = math.atan2(R[2, 1] / math.cos(yaw2), R[2, 2] / math.cos(yaw2))  # Pitch (X-axis rotation)
        
        roll2 = - math.atan2(R[1, 0] / math.cos(yaw2), R[0, 0] / math.cos(yaw2))   # Roll (Z-axis rotation)
    else:
        roll2 = 0  # Roll is zero
        if R[2, 0] == -1:
            yaw2 = math.pi / 2
            pitch2 = math.atan2(R[0, 1], R[0, 2])
        else:
            yaw2 = -math.pi / 2
            pitch2 = math.atan2(-R[0, 1], -R[0, 2])
    
    # Convert yaw to -90 to 90 degrees range
    if roll2 > math.pi / 2:
        roll2 -= math.pi  # Adjust yaw to the range -90 to 90 degrees
    elif roll2 < -math.pi / 2:
        roll2 += math.pi

    return np.array([pitch2, yaw2, roll2])


def generate_objp(checkerboard, scale):
    """
    Generate the 3D object points for a given checkerboard size and scale.

    Args:
        checkerboard (tuple): Checkerboard inner corners as (rows, cols).
        scale (float): Scale of the checkerboard squares.

    Returns:
        numpy.ndarray: 3D object points.
    """
    rows, cols = checkerboard
    objp = np.zeros((rows * cols, 3), np.float32)  # Efficiently allocate 3D points
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * scale
    return objp

def initialize_camera(fps=5, exposure_time=20000):
    """
    Initialize the camera with user-defined settings.

    Args:
        fps (int): Frames per second.
        exposure_time (int): Exposure time in microseconds.

    Returns:
        pylon.InstantCamera: Initialized camera object.
    """
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.Width.Value = camera.Width.Max
    camera.Height.Value = camera.Height.Max
    camera.OffsetX.Value = 0
    camera.OffsetY.Value = 0
    camera.AcquisitionFrameRateEnable = True
    camera.AcquisitionFrameRate = fps
    camera.ExposureAuto.SetValue('Off')
    camera.ExposureTime.SetValue(exposure_time)
    return camera

def load_calibration_data(filepath, img_size):
    """
    Load calibration data from a file and compute the new camera matrix.

    Args:
        filepath (str): Path to the calibration data file.
        img_size (tuple): Size of the images as (width, height).

    Returns:
        dict: Calibration data containing mtx, dist, mapx, mapy.
    """
    data = np.load(filepath)
    mtx = data['mtx']
    dist = data['dist']
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, img_size, 1, img_size)
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, img_size, cv2.CV_32FC1)
    return {'mtx': mtx, 'dist': dist, 'mapx': mapx, 'mapy': mapy, 'newcameramtx': newcameramtx}

def plot_to_image(rx, ry, rz, tx, ty, tz, ind_range):
    """

    Draw and Convert Matplotlib plots to an OpenCV-compatible image.

    Args:
        rx, ry, rz, tx, ty, tz (list): Rotation and translation data.
        ind_range (int): Number of data points to display.

    Returns:
        numpy.ndarray: Rendered plot as a BGR image.
    """
    fig, axes = plt.subplots(2,3, figsize=(16, 8))
    canvas = FigureCanvas(fig)

    # Plot the data
    for data, ax in zip([rx, ry, rz, tx, ty, tz], axes.flat):
        ax.plot(data[-ind_range:] if data else [0])
        ax.set_xlim(0, ind_range)
        ax.set_ylim(min(data[-ind_range:]) if data else 0, max(data[-ind_range:]) if data else 1)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4f}")) 

    # labels
    axes[0, 0].set_title('X-Rotation, Pitch')
    axes[0, 1].set_title('Y-Rotation, Yaw')
    axes[0, 2].set_title('Z-Rotation, Roll')
    axes[1, 0].set_title('X-Translation')
    axes[1, 1].set_title('Y-Translation')
    axes[1, 2].set_title('Z-Translation')

    plt.tight_layout()

    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# ----------------------- Main Processing Loop -----------------------

def process_camera_feed(camera, calib_data, objp, checkerboard, ind_range, update_interval):
    """
    Process the camera feed and visualize rotation and translation data.

    Args:
        camera (pylon.InstantCamera): Initialized camera object.
        calib_data (dict): Calibration data containing mtx, dist, mapx, mapy.
        objp (numpy.ndarray): 3D object points.
        checkerboard (tuple): Checkerboard size (rows, cols).
        ind_range (int): Number of frames to display in the plots.
        update_interval (int): Interval for updating the plots.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rx, ry, rz, tx, ty, tz = [], [], [], [], [], []
    prev_first_corner = None
    threshold = 50
    frame_count = 0
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    is_running = True
    while camera.IsGrabbing() and is_running:
        grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            img = grabResult.Array
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            dst = cv2.remap(img, calib_data['mapx'], calib_data['mapy'], cv2.INTER_LINEAR)
            dst_color = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
            ret, corners = cv2.findChessboardCorners(dst, checkerboard, cv2.CALIB_CB_FAST_CHECK)
            if ret:
                corners2 = cv2.cornerSubPix(dst, corners, (3, 3), (-1, -1), criteria)
                if prev_first_corner is not None:
                    diff = np.linalg.norm(corners2[0] - prev_first_corner)
                    if diff > threshold:
                        corners2 = np.flip(corners2, axis=0)

                prev_first_corner = corners2[0].copy()
                cv2.drawChessboardCorners(dst_color, checkerboard, corners2, ret)

                ret2, rvecs, tvecs = cv2.solvePnP(objp, corners2, calib_data['newcameramtx'], None)
                if ret2:
                    R = cv2.Rodrigues(rvecs)[0]
                    rvec = np.rad2deg(rotationMatrixToEulerAngles(R))
                    rx.append(rvec[0])
                    ry.append(rvec[1])
                    rz.append(rvec[2])
                    tx.append(-tvecs[0])
                    ty.append(-tvecs[1])
                    tz.append(-tvecs[2])

                if frame_count % update_interval == 0:
                    plot_img = plot_to_image(rx, ry, rz, tx, ty, tz, ind_range)
                    cv2.imshow('Plots', plot_img)
                    # cv2.imwrite('PoseEstimator/pose_estimation_plot.png', plot_img)
                    dst_color = plotXYZ(dst_color, rvecs, tvecs, calib_data['newcameramtx'])
                    # cv2.imwrite('PoseEstimator/pose_estimation.png', dst_color)
                    cv2.imshow('Chessboard Detection', dst_color)       
                    key= cv2.waitKey(1) & 0xFF
                    if key == 27:
                        is_running = False
                        break
            frame_count += 1




        grabResult.Release()

    cv2.destroyAllWindows()

# ----------------------- Main -----------------------

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Camera pose estimator using a checkerboard.")
    parser.add_argument("--checkerboard", type=int, nargs=2, required=True, default=[5, 7],
                        help="Checkerboard size as two integers: rows cols (e.g., 5 7).")
    parser.add_argument("--square_size", type=float, required=True, default=10,
                        help="Size of a square on the chessboard in mm or other units (e.g., 10).")
    parser.add_argument("--exposure_time", type=int, default=20000,
                        help="Exposure time in microseconds (default: 20000).")
    args = parser.parse_args()

    checkerboard = tuple(args.checkerboard)
    exposure_time = args.exposure_time
    square_size = args.square_size

    camera = initialize_camera(fps=100, exposure_time=exposure_time)

    calib_data = load_calibration_data('PoseEstimator/calib.npz', img_size= (camera.Width.Value, camera.Height.Value))

    objp = generate_objp(checkerboard, scale=square_size)
    try:
        process_camera_feed(
            camera,
            calib_data,
            objp,
            checkerboard,
            ind_range=50,
            update_interval=1
        )
    finally:
        camera.Close()
