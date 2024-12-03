import argparse
import cv2
import numpy as np
import os
from pypylon import pylon

# ----------------------- Utility Functions -----------------------


def initialize_camera(fps=100, exposure_time=20000):
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
    camera.ExposureAuto.SetValue("Off")
    camera.ExposureTime.SetValue(exposure_time)
    return camera


def prepare_object_points(checkerboard, square_size):
    """
    Prepare 3D object points for the given checkerboard size.

    Args:
        checkerboard (tuple): Checkerboard size (rows, cols).
        square_size (float): Size of each square in real-world units.

    Returns:
        numpy.ndarray: Object points for the chessboard.
    """
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0 : checkerboard[0], 0 : checkerboard[1]].T.reshape(-1, 2)
        * square_size
    )
    return objp


def capture_chessboard_images(
    camera, checkerboard, objp, save_dir="PoseEstimator/calibration_images"
):
    """
    Capture images with detected chessboard corners.

    Args:
        camera (pylon.InstantCamera): Initialized camera object.
        checkerboard (tuple): Checkerboard size (rows, cols).
        objp (numpy.ndarray): 3D object points for the chessboard.
        save_dir (str): Directory to save calibration images.

    Returns:
        tuple: Object points list, image points list, saved images list.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints, imgpoints, saved_images = [], [], []
    frame_count = 0

    print("Press 'Spacebar' to save images and corners.")
    print("Press 'ESC' to stop capturing and proceed to calibration.")
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    print("Capturing images...")
    print(camera.IsGrabbing())
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        print("Grabbed frame {}".format(frame_count))
        if grabResult.GrabSucceeded():
            img = grabResult.Array
            # if grayscale image, convert to BGR
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)

            # current status text
            current_status = (
                f"Frame: {frame_count}, Corners: {len(corners2) if ret else 0}"
            )
            instruction = "Press 'Spacebar' to save image and corners, 'ESC' to exit capture mode."

            # Display the current status on left top corner
            img = cv2.putText(
                img,
                current_status,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # Display the instruction on right bottom corner
            img = cv2.putText(
                img,
                instruction,
                (img.shape[1] - 600, img.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            # Display the image
            cv2.imshow("Chessboard Detection", img)

            # Handle keyboard inputs
            key = cv2.waitKey(1)
            if key == 32:  # Spacebar
                image_name = os.path.join(save_dir, f"image_{frame_count}.png")
                cv2.imwrite(image_name, img)
                saved_images.append(image_name)

                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    print(f"Saved: {image_name} and corners.")
                else:
                    print(f"Saved: {image_name}, but no corners detected.")
                frame_count += 1

            elif key == 27:  # ESC
                print("Exiting capture mode...")
                break

        grabResult.Release()

    cv2.destroyAllWindows()
    return objpoints, imgpoints, saved_images


def calibrate_camera(
    objpoints, imgpoints, image_shape, output_file="PoseEstimator/calib.npz"
):
    """
    Perform camera calibration and save the results.

    Args:
        objpoints (list): List of 3D object points.
        imgpoints (list): List of 2D image points.
        image_shape (tuple): Shape of the images used for calibration.
        output_file (str): File to save calibration results.

    Returns:
        tuple: Camera matrix and distortion coefficients.
    """
    print("Performing camera calibration...")
    if len(objpoints) > 0 and len(imgpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_shape[::-1], None, None
        )
        if ret:
            print(
                f"Calibration successful!\nCamera matrix:\n{mtx}\nDistortion coefficients:\n{dist}"
            )
            np.savez(output_file, mtx=mtx, dist=dist)
            print(f"Calibration data saved to '{output_file}'.")
            return mtx, dist
        else:
            raise RuntimeError("Calibration failed due to insufficient data.")
    else:
        raise ValueError("Not enough valid data for calibration.")


def load_calibration_data(calib_file):
    """
    Load camera calibration data from a file.

    Args:
        calib_file (str): File containing camera calibration data.

    Returns:
        dict: Camera matrix and distortion coefficients.
    """
    data = np.load(calib_file)
    mtx, dist = data["mtx"], data["dist"]
    return {"mtx": mtx, "dist": dist}


def undistort_image(img, calib_data):
    """
    Undistort the input image using camera calibration data.

    Args:
        img (numpy.ndarray): Input image to undistort.
        calib_data (dict): Camera calibration data containing mtx and dist.

    Returns:
        numpy.ndarray: Undistorted image.
    """
    return cv2.undistort(img, calib_data["mtx"], calib_data["dist"])


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Camera Calibration Tool")
    parser.add_argument(
        "--checkerboard",
        type=int,
        nargs=2,
        required=True,
        help="Checkerboard size as two integers: rows cols (e.g., 5 7).",
    )
    parser.add_argument(
        "--square_size",
        type=float,
        required=True,
        help="Size of a square on the chessboard in mm or other units (e.g., 10).",
    )
    parser.add_argument(
        "--exposure_time",
        type=int,
        default=20000,
        help="Exposure time in microseconds (default: 20000).",
    )
    args = parser.parse_args()

    checkerboard = tuple(args.checkerboard)
    exposure_time = args.exposure_time
    square_size = args.square_size

    # Initialize camera
    camera = initialize_camera(exposure_time=exposure_time)

    try:
        # Prepare 3D object
        objp = prepare_object_points(checkerboard, square_size)

        # Capture images and corners
        objpoints, imgpoints, saved_images = capture_chessboard_images(
            camera, checkerboard, objp
        )

        if saved_images:
            # Perform calibration
            calibrate_camera(
                objpoints,
                imgpoints,
                cv2.imread(saved_images[0], cv2.IMREAD_GRAYSCALE).shape,
            )

    finally:

        camera.Close()
        cv2.destroyAllWindows()
