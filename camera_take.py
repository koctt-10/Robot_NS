import cv2

link = ''
cap = cv2.VideoCapture(link)
width = 400
height = 300
dim = (width, height)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()