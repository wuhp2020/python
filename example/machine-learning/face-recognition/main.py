'''
简单易用的 python 人脸识别库
python3.8 -m pip install opencv-python -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install dlib -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install face_recognition -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install imutils -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install pandas -i https://mirrors.aliyun.com/pypi/simple
python3.8 -m pip install scikit-image -i https://mirrors.aliyun.com/pypi/simple
'''

# 摄像头实时人脸识别
import os
import dlib  # 人脸处理的库 Dlib
import csv  # 存入表格
import time
import sys
import numpy as np  # 数据处理的库 numpy
import cv2  # 图像处理的库 OpenCv
import pandas as pd  # 数据处理的库 Pandas
from imutils import face_utils


# 眼长宽比例
def eye_aspect_ratio(eye):
    # (|e1-e5|+|e2-e4|) / (2|e0-e3|)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# 嘴长宽比例
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[1] - mouth[7])  # 61, 67
    B = np.linalg.norm(mouth[3] - mouth[5])  # 63, 65
    C = np.linalg.norm(mouth[0] - mouth[4])  # 60, 64
    mar = (A + B) / (2.0 * C)
    return mar


# 人脸识别模型，提取128D的特征矢量
# face recognition model, the object maps human faces into 128D vectors
# Refer this tutorial: http://dlib.net/python/index.html#dlib.face_recognition_model_v1
facerec = dlib.face_recognition_model_v1(
    "./model/dlib_face_recognition_resnet_model_v1.dat")


# 计算两个128D向量间的欧式距离
# compute the e-distance between two 128D features
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist


# 处理存放所有人脸特征的 csv
path_features_known_csv = "./features/features_all.csv"
csv_rd = pd.read_csv(path_features_known_csv, header=None)

# 用来存放所有录入人脸特征的数组
# the array to save the features of faces in the database
features_known_arr = []

# 读取已知人脸数据
# print known faces
for i in range(csv_rd.shape[0]):
    features_someone_arr = []
    for j in range(0, len(csv_rd.loc[i, :])):
        features_someone_arr.append(csv_rd.loc[i, :][j])
    features_known_arr.append(features_someone_arr)
print("Faces in Database：", len(features_known_arr))

# Dlib 检测器和预测器
# The detector and predictor will be used
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')

# 创建 cv2 摄像头对象
# cv2.VideoCapture(0) to use the default camera of PC,
# and you can use local video name by use cv2.VideoCapture(filename)
cap = cv2.VideoCapture(0)

# cap.set(propId, value)
# 设置视频参数，propId 设置的视频参数，value 设置的参数值
cap.set(3, 480)

# cap.isOpened() 返回 true/false 检查初始化是否成功
# when the camera is open

# 眼长宽比例值
EAR_THRESH = 0.15
EAR_CONSEC_FRAMES_MIN = 2
EAR_CONSEC_FRAMES_MAX = 5  # 当EAR小于阈值时，接连多少帧一定发生眨眼动作

# 嘴长宽比例值
MAR_THRESH = 0.3

# 初始化眨眼的连续帧数
blink_counter = 0
# 初始化眨眼次数总数
blink_total = 0
# 初始化张嘴次数
mouth_total = 0
# 初始化张嘴状态为闭嘴
mouth_status_open = 0

print("[INFO] loading facial landmark predictor...")
# 人脸检测器
detector = dlib.get_frontal_face_detector()
# 特征点检测器
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
# 获取左眼的特征点
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# 获取右眼的特征点
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# 获取嘴巴特征点
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

print("[INFO] starting video stream thread...")


while cap.isOpened():

    flag, img_rd = cap.read()
    kk = cv2.waitKey(1)

    # 取灰度
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    # 人脸数 faces
    faces = detector(img_gray, 0)

    # 待会要写的字体 font to write later
    font = cv2.FONT_HERSHEY_COMPLEX

    # 存储当前摄像头中捕获到的所有人脸的坐标/名字
    # the list to save the positions and names of current faces captured
    pos_namelist = []
    name_namelist = []

    # 按下 q 键退出
    # press 'q' to exit
    if kk == ord('q'):
        break
    else:
        # 检测到人脸 when face detected
        if len(faces) != 0:
            # 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
            # get the features captured and save into features_cap_arr
            features_cap_arr = []
            for i in range(len(faces)):
                shape = predictor(img_rd, faces[i])
                features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))

                # 遍历捕获到的图像中所有的人脸
            # traversal all the faces in the database
            for k in range(len(faces)):
                print("##### camera person", k + 1, "#####")
                # 让人名跟随在矩形框的下方
                # 确定人名的位置坐标
                # 先默认所有人不认识，是 unknown
                # set the default names of faces with "unknown"
                name_namelist.append("unknown")

                # 每个捕获人脸的名字坐标 the positions of faces captured
                pos_namelist.append(
                    tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                # 对于某张人脸，遍历所有存储的人脸特征
                # for every faces detected, compare the faces in the database
                e_distance_list = []
                for i in range(len(features_known_arr)):
                    # 如果 person_X 数据不为空
                    if str(features_known_arr[i][0]) != '0.0':
                        print("with person", str(i + 1), "the e distance: ", end='')
                        e_distance_tmp = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                        print(e_distance_tmp)
                        e_distance_list.append(e_distance_tmp)
                    else:
                        # 空数据 person_X
                        e_distance_list.append(999999999)
                        # 找出最接近的一个人脸数据是第几个
                # Find the one with minimum e distance
                similar_person_num = e_distance_list.index(min(e_distance_list))
                print("Minimum e distance with person", int(similar_person_num) + 1)

                # 计算人脸识别特征与数据集特征的欧氏距离
                # 距离小于0.4则标出为可识别人物

                ########################################
                shape = predictor(img_gray, faces[0])  # 保存68个特征点坐标的<class 'dlib.dlib.full_object_detection'>对象
                shape = face_utils.shape_to_np(shape)  # 将shape转换为numpy数组，数组中每个元素为特征点坐标

                left_eye = shape[lStart:lEnd]  # 取出左眼对应的特征点
                right_eye = shape[rStart:rEnd]  # 取出右眼对应的特征点
                left_ear = eye_aspect_ratio(left_eye)  # 计算左眼EAR
                right_ear = eye_aspect_ratio(right_eye)  # 计算右眼EAR
                ear = (left_ear + right_ear) / 2.0  # 求左右眼EAR的均值

                inner_mouth = shape[mStart:mEnd]  # 取出嘴巴对应的特征点
                mar = mouth_aspect_ratio(inner_mouth)  # 求嘴巴mar的均值
                left_eye_hull = cv2.convexHull(left_eye)  # 寻找左眼轮廓
                right_eye_hull = cv2.convexHull(right_eye)  # 寻找右眼轮廓
                mouth_hull = cv2.convexHull(inner_mouth)  # 寻找内嘴巴轮廓
                cv2.drawContours(img_rd, [left_eye_hull], -1, (0, 255, 0), 1)  # 绘制左眼轮廓
                cv2.drawContours(img_rd, [right_eye_hull], -1, (0, 255, 0), 1)  # 绘制右眼轮廓
                cv2.drawContours(img_rd, [mouth_hull], -1, (0, 255, 0), 1)  # 绘制嘴巴轮廓

                # EAR低于阈值，有可能发生眨眼，眨眼连续帧数加一次
                if ear < EAR_THRESH:
                    blink_counter += 1

                # EAR高于阈值，判断前面连续闭眼帧数，如果在合理范围内，说明发生眨眼
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if EAR_CONSEC_FRAMES_MIN <= blink_counter <= EAR_CONSEC_FRAMES_MAX:
                        blink_total += 1
                    blink_counter = 0
                # 通过张、闭来判断一次张嘴动作
                if mar > MAR_THRESH:
                    mouth_status_open = 1
                else:
                    if mouth_status_open:
                        mouth_total += 1
                    mouth_status_open = 0

                cv2.putText(img_rd, "Blinks: {}".format(blink_total), (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img_rd, "Mouth: {}".format(mouth_total),
                            (130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img_rd, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img_rd, "MAR: {:.2f}".format(mar), (450, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



                if min(e_distance_list) < 0.4 and (blink_total >= 1 or mouth_total >= 1):
                    print("检测到眨眼睛")
                    print("检测到张嘴巴")
                    # 这里可以修改摄像头中标出的人名
                    # Here you can modify the names shown on the camera
                    # 1、遍历文件夹目录
                    folder_name = './images/'
                    # 最接近的人脸
                    sum = similar_person_num + 1
                    key_id = 1  # 从第一个人脸数据文件夹进行对比
                    # 获取文件夹中的文件名:1wang、2zhou、3...
                    file_names = os.listdir(folder_name)
                    for name in file_names:
                        # print(name+'->'+str(key_id))
                        if sum == key_id:
                            # winsound.Beep(300,500)# 响铃：300频率，500持续时间
                            name_namelist[k] = name  # 人名删去第一个数字（用于视频输出标识）
                        key_id += 1
                        # 播放欢迎光临音效
                    # playsound('D:/myworkspace/JupyterNotebook/People/music/welcome.wav')
                    # print("May be person "+str(int(similar_person_num)+1))
                    # -----------筛选出人脸并保存到visitor文件夹------------
                    for i, d in enumerate(faces):
                        x1 = d.top() if d.top() > 0 else 0
                        y1 = d.bottom() if d.bottom() > 0 else 0
                        x2 = d.left() if d.left() > 0 else 0
                        y2 = d.right() if d.right() > 0 else 0
                        face = img_rd[x1:y1, x2:y2]
                        size = 64
                        face = cv2.resize(face, (size, size))
                        # 要存储visitor人脸图像文件的路径
                        path_visitors_save_dir = "./unknownimages"
                        # 存储格式：2019-06-24-14-33-40wang.jpg
                        now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                        save_name = str(now_time) + str(name_namelist[k]) + '.jpg'
                        # print(save_name)
                        # 本次图片保存的完整url
                        save_path = path_visitors_save_dir + '/' + save_name
                        # 遍历visitor文件夹所有文件名
                        visitor_names = os.listdir(path_visitors_save_dir)
                        visitor_name = ''
                        for name in visitor_names:
                            # 名字切片到分钟数：2019-06-26-11-33-00wangyu.jpg
                            visitor_name = (name[0:16] + '-00' + name[19:])
                            # print(visitor_name)
                        visitor_save = (save_name[0:16] + '-00' + save_name[19:])
                        # print(visitor_save)
                        # 一分钟之内重复的人名不保存
                        if visitor_save != visitor_name:
                            # cv2.imwrite(save_path, face)
                            print(
                                '新存储：' + path_visitors_save_dir + '/' + str(now_time) + str(name_namelist[k]) + '.jpg')
                        else:
                            print('重复，未保存！')

                else:
                    blink_total = 0
                    mouth_total = 0
                    # 播放无法识别音效
                    # playsound('D:/myworkspace/JupyterNotebook/People/music/sorry.wav')
                    print("Unknown person")
                    # -----保存图片-------
                    # -----------筛选出人脸并保存到visitor文件夹------------
                    for i, d in enumerate(faces):
                        x1 = d.top() if d.top() > 0 else 0
                        y1 = d.bottom() if d.bottom() > 0 else 0
                        x2 = d.left() if d.left() > 0 else 0
                        y2 = d.right() if d.right() > 0 else 0
                        face = img_rd[x1:y1, x2:y2]
                        size = 64
                        face = cv2.resize(face, (size, size))
                        # 要存储visitor-》unknown人脸图像文件的路径
                        path_visitors_save_dir = "./unknownimages"
                        # 存储格式：2019-06-24-14-33-40unknown.jpg
                        now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                        # print(save_name)
                        # 本次图片保存的完整url
                        save_path = path_visitors_save_dir + '/' + str(now_time) + 'unknown.jpg'
                        # cv2.imwrite(save_path, face)
                        print('新存储：' + path_visitors_save_dir + '/' + str(now_time) + 'unknown.jpg')

                        # 矩形框
                # draw rectangle
                for kk, d in enumerate(faces):
                    # 绘制矩形框
                    cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
                print('\n')

                # 在人脸框下面写人脸名字
            # write names under rectangle
            for i in range(len(faces)):
                cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

    print("Faces in camera now:", name_namelist, "\n")

    # cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Visitors: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # 窗口显示 show with opencv
    cv2.imshow("camera", img_rd)

# 释放摄像头 release camera
cap.release()

# 删除建立的窗口 delete all the windows
cv2.destroyAllWindows()