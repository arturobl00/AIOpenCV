import cv2
import mediapipe as mp

# utils for drawing on image
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# mediapipe pose model
mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5)

image = cv2.imread("ronaldo.jpg")
#convert image to RGB (just for input to model)
image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# get results using mediapipe
results = pose.process(image_input)

if not results.pose_landmarks:
    print("no results found")
else:
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

# write image to storage
cv2.imwrite("./ronaldo-processed.jpg", image)