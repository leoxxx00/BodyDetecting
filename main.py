import cv2
import mediapipe as mp

def main():
    # Initialize Mediapipe Pose and Hands modules
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    pose = mp_pose.Pose()
    hands = mp_hands.Hands()

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process body pose
        results_pose = pose.process(rgb_frame)
        if results_pose.pose_landmarks:
            # Draw landmarks on the frame if needed
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Process hand pose
        results_hands = hands.process(rgb_frame)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Draw hand landmarks on the frame if needed
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow('AI Body and Hand Pose Detection', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
