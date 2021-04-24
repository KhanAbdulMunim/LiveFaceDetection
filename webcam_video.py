import cv2 as cv

vid = cv.VideoCapture(0)
haar_cascade = cv.CascadeClassifier("haar_face.xml")

while True:
    # define a video capture object

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # print(faces_rect)
    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv.imshow("frame", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

vid.release()
cv.destroyAllWindows()
