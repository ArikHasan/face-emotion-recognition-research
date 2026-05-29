import cv2
from fer import FER


def main():
    """
    Real-Time Face Emotion Recognition System
    Author: Mahmudul Hasan
    Technology: Python, OpenCV, FER, TensorFlow
    """

    print("Starting Real-Time Face Emotion Recognition System...")
    print("Press 'q' to quit the application.")

    # Load FER emotion detector
    detector = FER(mtcnn=False)

    # Open webcam
    camera = cv2.VideoCapture(0)

    # Low configuration laptop friendly resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        print("Try changing cv2.VideoCapture(0) to cv2.VideoCapture(1).")
        return

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame from webcam.")
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # Detect emotions from current frame
        results = detector.detect_emotions(frame)

        for result in results:
            x, y, w, h = result["box"]
            emotions = result["emotions"]

            # Get emotion with highest confidence
            top_emotion = max(emotions, key=emotions.get)
            confidence = emotions[top_emotion] * 100

            label = f"{top_emotion.upper()} ({confidence:.1f}%)"

            # Draw face rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Draw label background
            cv2.rectangle(
                frame,
                (x, y - 35),
                (x + w, y),
                (0, 255, 0),
                -1
            )

            # Show emotion label
            cv2.putText(
                frame,
                label,
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

            # Show emotion scores on left side
            start_y = 30
            cv2.putText(
                frame,
                "Emotion Scores:",
                (10, start_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            for index, (emotion, score) in enumerate(emotions.items()):
                score_text = f"{emotion}: {score * 100:.1f}%"
                cv2.putText(
                    frame,
                    score_text,
                    (10, start_y + 30 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        # Display output
        cv2.imshow("Real-Time Face Emotion Recognition", frame)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Application closed successfully.")


if __name__ == "__main__":
    main()
