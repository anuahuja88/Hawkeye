import numpy as np
import cv2
from collections import Counter
from ultralytics import YOLO
import os

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

def process_video(video_file, model):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    y_history = []
    detected_landings = []
    shuttle_num = 0
    shuttle_pos_dict = {}
    shuttle_position_prev = []

    # Initialize the back line information
    # back_line_vector, point, m, c, vertical_x = Hough_line_probabalistic(video_file)

    # Create directory to save landing frames
    output_dir = 'shuttle_landings'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                shuttle_positions = [box.xyxy[0].cpu().numpy() for box in boxes if box.cls == 0]  # Assuming class index 0 for shuttle
                
                if len(shuttle_positions) > 0:
                    shuttle_positions = np.array(shuttle_positions)
                    if shuttle_num not in shuttle_pos_dict:
                        shuttle_pos_dict[shuttle_num] = [shuttle_positions]     
                    else:
                        shuttle_pos_dict[shuttle_num].append(shuttle_positions)
                    shuttle_position_prev = [shuttle_positions]
                    if len(shuttle_position_prev) == 0:
                        shuttle_num += 1
                else:
                        shuttle_position_prev = []
        
            
            #     if shuttle_positions:
            #         shuttle_positions = np.array(shuttle_positions)
            #         lowest_shuttle = shuttle_positions[np.argmin(shuttle_positions[:, 3])]  # y_max position
            #         y_history.append((frame_count, lowest_shuttle[3]))

            #         # Check for bounce: current frame y > previous frame y (shuttle going up)
            #         if len(y_history) > 1 and y_history[-1][1] > y_history[-2][1]:
            #             landing_frame = y_history[-2][0]
            #             landing_position = y_history[-2][1]
            #             detected_landings.append((landing_frame, landing_position))
                        
            #             # Save the frame with shuttle landing
            #             cap.set(cv2.CAP_PROP_POS_FRAMES, landing_frame)
            #             ret, landing_frame_img = cap.read()
            #             if ret:
            #                 frame_filename = os.path.join(output_dir, f"landing_frame_{landing_frame}.png")
            #                 cv2.imwrite(frame_filename, landing_frame_img)
            #                 print(f"Landing frame saved: {frame_filename}")
                        
            #             y_history = []  # Reset history for the next shuttle

            # frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(shuttle_num)
    return shuttle_pos_dict

def main():
    model = YOLO('runs/detect/train/weights/best.pt')
    shuttle_pos_dict = process_video("Hawkeye/Videos/new1.mp4", model)
if __name__ == "__main__":
    main()
