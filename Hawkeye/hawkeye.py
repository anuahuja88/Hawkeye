import numpy as np
import cv2
import os
from collections import Counter
from ultralytics import YOLO

def compute_line_equation(vector, point):
    """
    Compute the line equation y = mx + c from the vector and a point.
    """
    (dx, dy) = vector
    (x1, y1) = point
    
    if dx == 0:
        # Vertical line case: x = x1
        return None, None, x1
    
    # Calculate the slope (m)
    m = dy / dx
    
    # Calculate the y-intercept (c) using y = mx + c -> c = y - mx
    c = y1 - m * x1
    
    return m, c, None

def Hough_line_probabalistic(video_file):
    vectorLists = []
    linePoints = []
    cap = cv2.VideoCapture(video_file)
    back_line_vector = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(grey, (5, 5), 0)
        edges = cv2.Canny(img_blur, 50, 150, apertureSize=3)

        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                line_length = np.sqrt((l[2] - l[0])**2 + (l[3] - l[1])**2)
                orientation = abs(l[3] - l[1]) / abs(l[2] - l[0] + 1e-5)

                if line_length > 200 and orientation > 0.1:  # Adjust these thresholds as needed
                    cv2.line(frame, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 2, cv2.LINE_AA)
                    back_line_vector = (l[2] - l[0], l[3] - l[1])
                    vectorLists.append(back_line_vector)
                    linePoints.append((l[0], l[1], l[2], l[3]))
                    print(f"Back line detected: Point 1: ({l[0]}, {l[1]}), Point 2: ({l[2]}, {l[3]})")
                    print(f"Back line vector: {back_line_vector}")

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Find the most frequent vector
    if vectorLists:
        vector_counts = Counter(vectorLists)
        best_line_vector, best_line_count = vector_counts.most_common(1)[0]
        
        # Find the first occurrence of this vector to get its points
        for points in linePoints:
            (x1, y1, x2, y2) = points
            if (x2 - x1, y2 - y1) == best_line_vector:
                point = (x1, y1)
                break
        
        m, c, vertical_x = compute_line_equation(best_line_vector, point)
        if m is not None:
            print(f"Line equation: y = {m}x + {c}")
        else:
            print(f"Vertical line at x = {vertical_x}")
        
        return best_line_vector
    else:
        return None

def main():
    # vector = Hough_line_probabalistic("Hawkeye/Videos/new1.mp4")
    # if vector is not None:
    #     print(f"Most frequent back line vector: {vector}")
    # else:
    #     print("No back line detected")

    model = YOLO('runs/detect/train/weights/best.pt')
    model.predict("Hawkeye/Videos/new1.mp4", save = True, show = True)

if __name__ == "__main__":
    main()
