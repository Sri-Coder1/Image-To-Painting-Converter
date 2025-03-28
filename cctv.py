import cv2  #install opencv-python through terminal
import time
import os
import win32gui  # type: ignore
import win32con  # type: ignore

# Minimize the current window
def minimizeWindow():
    window = win32gui.GetForegroundWindow()
    win32gui.ShowWindow(window, win32con.SW_MINIMIZE)

# Detect faces and draw rectangles around them
def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Main CCTV function to capture and save video
def cctv():
    try:
        video = cv2.VideoCapture(0)  # Open the default camera
        if not video.isOpened():
            raise Exception("Could not access the camera.")
        
        video.set(3, 1280)
        video.set(4, 720)
        print("Video resolution is set to 1280x720")

        date_time = time.strftime("recording_time_%H-%M-%S_date_%d-%m-%y")
        output_dir = r'E:\Ankith\projects\footages_'  # enter your file path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = os.path.join(output_dir, f'footages_{date_time}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))

        while video.isOpened():
            check, frame = video.read()
            if check:
                detect_faces(frame)
                cv2.imshow("CCTV Camera", frame)
                output.write(frame)

                key = cv2.waitKey(1)
                if key == 27:  # ESC key to exit
                    print("Video footage saved in:", filename)
                    break
                elif key == ord('m'):
                    minimizeWindow()  # Minimize the window if 'm' is pressed
            else:
                print("Cannot open the camera")
                break

        video.release()
        output.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")

# Input section
if __name__ == "__main__":
    print("************ Welcome to CCTV Software ************")
    cctv()  # Start the CCTV system
