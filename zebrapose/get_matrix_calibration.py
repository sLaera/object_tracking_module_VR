import numpy as np
import cv2 as cv
import time
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = []

# Open the camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
i = 0
while True:
    ret, frame = cap.read()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret and len(corners) > 0:
        objpoints.append(objp)

    if ret and len(corners) > 0:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(frame, (7, 6), corners2, ret)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            print("Matrice intrinseca della fotocamera:")
            print(mtx)
        time.sleep(1)
        
    cv.imshow('img', frame)

    if cv.waitKey(1) & 0xFF == ord('c'):
        images.append(frame.copy())
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            print("Matrice intrinseca della fotocamera:")
            print(mtx)
        time.sleep(1)

# Release the capture
cap.release()
cv.destroyAllWindows()
