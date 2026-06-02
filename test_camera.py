import cv2

print("Testing camera with CV2...")

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ Failed to open camera")
else:
    print("✅ Camera opened")
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite("test_capture.jpg", frame)
        print("✅ Image saved as test_capture.jpg")
        print("Check the TeddyBear folder for the photo")
    else:
        print("❌ Failed to capture frame")

print("Test finished")
