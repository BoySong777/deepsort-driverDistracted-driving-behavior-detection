# 1. 引入必要的库
import cv2
import matplotlib.pyplot as plt
import dlib

# 2.读取图片
# image = cv2.imread("3171314575.jpg")

# 打开摄像头
capture = cv2.VideoCapture(0)

# 3. 调用人脸检测器
detector = dlib.get_frontal_face_detector()

# 4.加载预测关键点模型
predictor = dlib.shape_predictor("../weights/shape_predictor_68_face_landmarks.dat")

while True:
    #读取每一帧照片
    ret, image = capture.read()
    # 5. 灰度转换
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 6.人脸检测
    faces = detector(gray, 1)

    # 7. 循环遍历每一张人脸，给人脸绘制矩形框和关键点
    for face in faces:
        # 8. 绘制人脸矩形框
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 3)

        # 9. 预测关键点
        shape = predictor(image, face)

        # 10. 获取关键点坐标
        for pt in  shape.parts():
            # 获取横纵坐标
            pt_position = (pt.x, pt.y)
            # 11 绘制关键点坐标
            cv2.circle(image, pt_position, 2, (0,0,255), -1)
    if cv2.waitKey(1)& 0xFF == ord("q"):
        break

    cv2.imshow("face detection", image)

# 12 显示
# plt.imshow(image)
# plt.axis("off")
# plt.show()

