import numpy as np
import cv2
import os

CHECKERBOARD_SIZE = (9, 6)

min_thres = 125
max_thres = 200
hough_thres = 100


def calibrate_iphone():
    # Termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space.
    imgpoints = [] # 2d points in image plane.

    for fname in os.listdir("Hawkeye/Calibration images"):
        print(fname)
        img = cv2.imread(f"Hawkeye/Calibration images/{fname}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img.shape)

        if img.shape == (2048, 1536, 3):
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

            if ret:
                objpoints.append(objp)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners_refined)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners_refined, ret)
                cv2.imshow('Corners Detected', img)
                cv2.waitKey(500)  # Wait for a moment to display the image

    cv2.destroyAllWindows()

    # Calibrate camera
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    cv2.destroyAllWindows()

    h, w = gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    np.savetxt('camera_matrix.npy', mtx)
    np.savetxt('distortion_coeff.npy', dist)
    np.savetxt('new_cam_matrix.npy', newcameramtx)
    return newcameramtx

def Hough_line_probabalistic(video_file):
    cap = cv2.VideoCapture(video_file)

    def change_min(x):
        global min_thres
        min_thres = x

    def change_max(x):
        global max_thres
        max_thres = x

    def change_hough(x):
        global hough_thres
        hough_thres = x

    cv2.namedWindow('Video')
    cv2.createTrackbar('Min Threshold', 'Video', min_thres, 255, change_min)
    cv2.createTrackbar('Max Threshold', 'Video', max_thres, 255, change_max)
    cv2.createTrackbar('Hough Threshold', 'Video', hough_thres, 255, change_hough)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.blur(grey, (3,3))

        edges = cv2.Canny(img_blur, 130, 250, None, 3)

        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 0, None, 50, 10)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                print(f"Point 1:{(l[0], l[1])}")
                print(f"Point 2:{(l[2], l[3])}")
                cv2.line(edges, (l[0], l[1]), (l[2], l[3]), (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('Video', edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Hough_line_standard("Hawkeye/Videos/first.mp4")
    Hough_line_probabalistic("Hawkeye/Videos/first.mp4")

if __name__ == "__main__":
    main()