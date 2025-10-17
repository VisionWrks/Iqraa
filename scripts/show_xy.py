import cv2

img = cv2.imread("result/lines_ink.png")

def show_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"x={x}, y={y}, BGR={img[y, x].tolist()}")

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", show_xy)
cv2.waitKey(0)
cv2.destroyAllWindows()
