import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from tkinter import filedialog, Button, Label, Tk, Frame
import matplotlib
matplotlib.use('TkAgg')

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    # Preprocessing the image for the model
    img_keras = image.load_img(img_path, target_size=target_size)
    img_tensor = image.img_to_array(img_keras)  # Convert to tensor
    img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension
    img_tensor /= 255.0  # Rescale to [0, 1]

    # Load the original image for displaying
    img_cv2 = cv2.imread(img_path)

    return img_cv2, img_tensor

def predict_image(model, img_path):
    original_img, img_tensor = load_and_preprocess_image(img_path)

    # Make prediction
    prediction = model.predict(img_tensor)
    confidence = prediction[0][0]
    print(f"Prediction: {'Face Detected' if confidence > 0.5 else 'No Face Detected'} (Confidence: {confidence})")

    if confidence > 0.5:
        # Use Haar Cascade to draw a rectangle around the face
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(original_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert image back to uint8 for display with cv2 and matplotlib
    original_img_uint8 = original_img.astype(np.uint8)

    # Display the image with matplotlib
    plt.imshow(cv2.cvtColor(original_img_uint8, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {'Face Detected' if confidence > 0.5 else 'No Face Detected'}")
    plt.show()

def browse_files():
    filename = filedialog.askopenfilename()
    if filename:
        predict_image(face_detection_model, filename)

def detect_face_live():
    stop_camera = False
    cap = cv2.VideoCapture(0)

    while not stop_camera:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frame_tensor = image.img_to_array(frame_resized)
        frame_tensor = np.expand_dims(frame_tensor, axis=0)
        frame_tensor /= 255.0

        # Predict using the model
        prediction = face_detection_model.predict(frame_tensor)
        confidence = prediction[0][0]

        if confidence > 0.5:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                label = "Face Detected"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            label = "No Face Detected"

        # Display the frame
        cv2.imshow("Face Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_faces_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frame_tensor = image.img_to_array(frame_resized)
        frame_tensor = np.expand_dims(frame_tensor, axis=0)
        frame_tensor /= 255.0

        # Predict using the model
        prediction = face_detection_model.predict(frame_tensor)
        confidence = prediction[0][0]

        if confidence > 0.5:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                label = "Face Detected"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            label = "No Face Detected"

        # Display the frame
        cv2.imshow("Face Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def browse_video_file():
    filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if filename:
        detect_faces_in_video(filename)

# Load the pre-trained model
face_detection_model = load_model('My_model.h5')

# Create GUI
root = Tk()
root.title("Face Detection Application")
root.geometry("600x400")
root.configure(bg="#2e3f4f")

frame = Frame(root, bg="#4e5d6c", bd=5, relief="groove")
frame.place(relx=0.5, rely=0.5, anchor="center")

button_browse = Button(frame, text="Browse Image", command=browse_files, width=20, height=2, bg="#e1e5ea", fg="black", font=("Arial", 12, "bold"), bd=3, relief="ridge")
button_browse.grid(row=0, column=0, padx=10, pady=10)

button_camera = Button(frame, text="Open Camera", command=detect_face_live, width=20, height=2, bg="#e1e5ea", fg="black", font=("Arial", 12, "bold"), bd=3, relief="ridge")
button_camera.grid(row=1, column=0, padx=10, pady=10)

button_video = Button(frame, text="Browse Video", command=browse_video_file, width=20, height=2, bg="#e1e5ea", fg="black", font=("Arial", 12, "bold"), bd=3, relief="ridge")
button_video.grid(row=2, column=0, padx=10, pady=10)

root.mainloop()
