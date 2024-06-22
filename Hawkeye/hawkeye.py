import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter
import time

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
    
    # Calculate the y-intercept (c) using y = mx + c -> c = y1 - m * x1
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
        
        return best_line_vector, point, m, c, vertical_x
    else:
        return None, None, None, None, None

def shuttle_positions(video_file, model):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    shuttle_num = 0
    shuttle_pos_dict = defaultdict(list)
    shuttle_position_prev = []
    frames_since_last_shuttle = 0
    frames_gap_threshold = 10  # Number of frames to wait before considering a new shuttle

    # Create directory to save landing frames
    output_dir = 'shuttle_landings'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        shuttle_positions = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                shuttle_positions = [box.xyxy[0].cpu().numpy() for box in boxes if box.cls == 0]  # Assuming class index 0 for shuttle

        if len(shuttle_positions) > 0:
            shuttle_positions = np.array(shuttle_positions)
            shuttle_pos_dict[shuttle_num].append((frame_count, shuttle_positions))
            shuttle_position_prev = shuttle_positions
            frames_since_last_shuttle = 0  # Reset counter
        else:
            frames_since_last_shuttle += 1
            if frames_since_last_shuttle > frames_gap_threshold:
                if len(shuttle_position_prev) > 0:
                    shuttle_num += 1
                    shuttle_position_prev = []

        # Display the frame with the shuttle count
        cv2.putText(frame, f'Shuttle Count: {shuttle_num}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    
    return shuttle_pos_dict

def get_landing_frame(video_file, shuttle_pos_dict):
    cap = cv2.VideoCapture(video_file)
    output_dir = 'shuttle_landings'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    landing_positions = {}  # Dictionary to store the landing positions

    for shuttle_num, positions in shuttle_pos_dict.items():
        max_y = float('-inf')
        landing_frame = None
        landing_bbox = None
        found_landing = False

        for i in range(1, len(positions)):
            prev_frame_count, prev_pos = positions[i-1]
            frame_count, pos = positions[i]

            prev_y_coords = prev_pos[:, 3]  # y_max position of previous frame
            y_coords = pos[:, 3]  # y_max position of current frame

            # Check if shuttle is rising after reaching the highest point (indicating a bounce)
            if np.max(y_coords) < np.max(prev_y_coords) and np.max(prev_y_coords) > max_y:
                max_y = np.max(prev_y_coords)
                landing_frame = prev_frame_count
                landing_bbox = prev_pos[np.argmax(prev_y_coords)]  # Store the bounding box of the landing shuttle
                found_landing = True
                break  # Found the landing frame, no need to check further

        if found_landing and landing_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, landing_frame)
            ret, frame = cap.read()
            if ret:
                frame_filename = os.path.join(output_dir, f"landing_frame_{shuttle_num}_{landing_frame}.png")
                cv2.imwrite(frame_filename, frame)
                print(f"Landing frame saved: {frame_filename}")
                landing_positions[shuttle_num] = (landing_frame, landing_bbox)  # Store the landing frame and bbox

    cap.release()
    cv2.destroyAllWindows()
    
    return landing_positions


def check_in_or_out(bbox, m, c, vertical_x):
    """
    Check if the shuttle is in or out based on its bounding box and the line equation.
    If the center of the bounding box is to the left of the line or on the line, it's in.
    If it's to the right, it's out.
    """
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2

    if vertical_x is not None:
        # Vertical line case
        if x_center <= vertical_x:
            return "Out"
        else:
            return "In"
    else:
        # General line case
        y_line = m * x_center + c
        if y_center <= y_line:
            return "Out"
        else:
            return "In"
        

def visualise_results(video_file, landing_positions, m, c, vertical_x, annotation_duration=1):
    cap = cv2.VideoCapture(video_file)
    result = ''
    num = 0
    
    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = int(1000 / fps)  # Duration per frame in milliseconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get current frame number
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # Check if current frame is a landing frame
        for shuttle_num, (landing_frame, bbox) in landing_positions.items():
            if frame_num == landing_frame:
                result = check_in_or_out(bbox, m, c, vertical_x)
                num += 1
                # Display frame with annotations and add delay for annotation
                cv2.putText(frame, f"Shuttle {num}: {result}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('Video with Annotations', frame)
                if cv2.waitKey(int(annotation_duration * 1000)) & 0xFF == ord('q'):
                    break

        # Display frame without annotation delay
        cv2.putText(frame, f"Shuttle {num}: {result}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Video with Annotations', frame)
        
        # Control playback speed to match the original video speed
        if cv2.waitKey(frame_duration) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
def main():
    video_file = "Hawkeye/Videos/new1.mp4"
    model = YOLO('runs/detect/train/weights/best.pt')
    shuttle_pos_dict = shuttle_positions(video_file, model)
    landing_positions = get_landing_frame(video_file, shuttle_pos_dict)
    best_line_vector, point, m, c, vertical_x = Hough_line_probabalistic(video_file)

    for shuttle_num, (frame, bbox) in landing_positions.items():
        result = check_in_or_out(bbox, m, c, vertical_x)
        print(f"Shuttle {shuttle_num} is {result}")

    visualise_results(video_file, landing_positions, m, c, vertical_x)

if __name__ == "__main__":
    main()
