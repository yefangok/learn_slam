'''
使用OpenCV实现ORB特征点提取
作者：知乎@Ai酱
代码地址：https://github.com/varyshare/easy_slam_tutorial/tree/master/feature_extract
'''
    
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [14.0, 7.0]

# 0. 读取图片
img = cv2.imread('../1.png',cv2.COLOR_BGR2GRAY)
training_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 1. 创建一个ORB检测器实例
orb = cv2.ORB_create()
# 2. 检测关键点
keypoint, descript = orb.detectAndCompute(training_gray,None)
# 3. 绘制关键点

# Draw the keypoints without size or orientation on one copy of the training image 
keypoint_without_size = cv2.drawKeypoints(img, keypoint, None, color = (0, 255, 0))

# Draw the keypoints with size and orientation on the other copy of the training image
keypoint_with_size = cv2.drawKeypoints(img, keypoint, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with the keypoints without size or orientation
plt.subplot(121)
plt.title('Keypoints Without Size or Orientation')
plt.imshow(keypoint_without_size)

# Display the image with the keypoints with size and orientation
plt.subplot(122)
plt.title('Keypoints With Size and Orientation')
plt.imshow(keypoint_with_size)
plt.show()

# Print the number of keypoints detected
print("\nNumber of keypoints Detected: ", len(keypoint))