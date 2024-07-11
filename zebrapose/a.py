import cv2

b_box = [68, 74, 74, 64]
test_img = cv2.imread("000000.png")
test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
# test_img = cv2.cvtColor(cv2.imread("1.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
x, y, w, h = b_box
cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("aaaaaa", test_img)
cv2.waitKey(0)
