import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Read input video
video_path = '/Users/aitorortunio/Downloads/Fisica/caida_talon.MOV'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(frame_width, frame_height, fps)

# Prepare output video
out = cv2.VideoWriter('/Users/aitorortunio/Downloads/Fisica/tracked_pie.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Prepare CSV file for landmark data
csv_file = open('/Users/aitorortunio/Downloads/Fisica/landmarks.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['FRAME', 'LEFT_ANKLE_X', 'LEFT_ANKLE_Y', 'LEFT_HEEL_X', 'LEFT_HEEL_Y', 'LEFT_FOOT_INDEX_X', 'LEFT_FOOT_INDEX_Y', 'LEFT_KNEE_X', 'LEFT_KNEE_Y'])

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect landmarks
    result = pose.process(rgb_frame)
    
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        
        # Extract the required landmarks
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value]
        left_foot_index = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]


        # Write the landmarks to the CSV file
        csv_writer.writerow([frame_index,
                             left_ankle.x, left_ankle.y,
                             left_heel.x, left_heel.y,
                             left_foot_index.x, left_foot_index.y,
                             left_knee.x, left_knee.y])

        # Convert normalized coordinates to pixel values
        h, w, _ = frame.shape
        right_ankle_coords = (int(left_ankle.x * w), int(left_ankle.y * h))
        right_heel_coords = (int(left_heel.x * w), int(left_heel.y * h))
        right_foot_index_coords = (int(left_foot_index.x * w), int(left_foot_index.y * h))
        right_knee_coords = (int(left_knee.x * w), int(left_knee.y * h))
        
        # Draw the landmarks
        cv2.circle(frame, right_ankle_coords, 15, (0, 255, 0), -1)
        cv2.circle(frame, right_heel_coords, 15, (0, 255, 0), -1)
        cv2.circle(frame, right_foot_index_coords, 15, (0, 255, 0), -1)
        cv2.circle(frame, right_knee_coords, 15, (0, 255, 0), -1)

        # Draw lines between the landmarks
        cv2.line(frame, right_ankle_coords, right_heel_coords, (255, 0, 0), 2)
        cv2.line(frame, right_heel_coords, right_foot_index_coords, (255, 0, 0), 2)
        cv2.line(frame, right_foot_index_coords, right_ankle_coords, (255, 0, 0), 2)
        cv2.line(frame, right_ankle_coords, right_knee_coords, (255, 0, 0), 2)
    
    # Write the frame to the output video
    out.write(frame)
    frame_index += 1

# Release resources
cap.release()
out.release()
csv_file.close()
pose.close()
cv2.destroyAllWindows()
