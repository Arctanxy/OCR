#coding:utf-8
#数据集扩增
import cv2
import math
import numpy as np
import xml.etree.ElementTree as ET
import os
 
def rotate_image(src, angle, scale=1):
    w = src.shape[1]
    h = src.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
 
    dst = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
    # 仿射变换
    return dst

def rotate_box(src, p1, p2, p3, p4, angle, scale = 1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    # 获取旋转后图像的长和宽
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # print('rot_mat=', rot_mat)# rot_mat是最终的旋转矩阵
    # point1 = np.dot(rot_mat, np.array([xmin, ymin, 1]))          #这种新画出的框大一圈
    # point2 = np.dot(rot_mat, np.array([xmax, ymin, 1]))
    # point3 = np.dot(rot_mat, np.array([xmax, ymax, 1]))
    # point4 = np.dot(rot_mat, np.array([xmin, ymax, 1]))
    point1 = np.dot(rot_mat, np.array([p1[0], p1[1], 1]))   # 获取原始矩形的四个中点，然后将这四个点转换到旋转后的坐标系下
    # print('point1=',point1)
    point2 = np.dot(rot_mat, np.array([p2[0], p2[1], 1]))
    # print('point2=', point2)
    point3 = np.dot(rot_mat, np.array([p3[0], p3[1], 1]))
    # print('point3=', point3)
    point4 = np.dot(rot_mat, np.array([p4[0], p4[1], 1]))
    # print('point4=', point4)
    # concat = np.vstack((point1, point2, point3, point4))            # 合并np.array
    return point1, point2, point3, point4
    # print('concat=', concat)
    # 改变array类型
    # concat = concat.astype(np.int32)
    # rx, ry, rw, rh = cv2.boundingRect(concat)                        #rx,ry,为新的外接框左上角坐标，rw为框宽度，rh为高度，新的xmax=rx+rw,新的ymax=ry+rh
    # return rx, ry, rw, rh




# 对应修改xml文件
def rotate_xml(src, xmin, ymin, xmax, ymax, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    # 获取旋转后图像的长和宽
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # print('rot_mat=', rot_mat)# rot_mat是最终的旋转矩阵
    # point1 = np.dot(rot_mat, np.array([xmin, ymin, 1]))          #这种新画出的框大一圈
    # point2 = np.dot(rot_mat, np.array([xmax, ymin, 1]))
    # point3 = np.dot(rot_mat, np.array([xmax, ymax, 1]))
    # point4 = np.dot(rot_mat, np.array([xmin, ymax, 1]))
    point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))   # 获取原始矩形的四个中点，然后将这四个点转换到旋转后的坐标系下
    # print('point1=',point1)
    point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
    # print('point2=', point2)
    point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
    # print('point3=', point3)
    point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
    # print('point4=', point4)
    concat = np.vstack((point1, point2, point3, point4))            # 合并np.array
    # print('concat=', concat)
    # 改变array类型
    concat = concat.astype(np.int32)
    rx, ry, rw, rh = cv2.boundingRect(concat)                        #rx,ry,为新的外接框左上角坐标，rw为框宽度，rh为高度，新的xmax=rx+rw,新的ymax=ry+rh
    return rx, ry, rw, rh

if __name__ == "__main__":
    '''使图像旋转15, 30, 45, 60, 75, 90, 105, 120度
    '''
    # imgpath = './images/train/'          #源图像路径
    imgpath = './train_example/'          #源图像路径
    xmlpath = './Annotations/'            #源图像所对应的xml文件路径
    rotated_imgpath = './train_example_out/'
    rotated_xmlpath = './Annotations_out/'
    if not (os.path.exists(rotated_imgpath) and os.path.exists(rotated_xmlpath)):
        os.mkdir(rotated_imgpath)
        os.mkdir(rotated_xmlpath)
    for angle in (15,30, 45, 60, 75, 90, 105, 120):
        for i in os.listdir(imgpath):
            a, b = os.path.splitext(i)                            #分离出文件名a
    
            img = cv2.imread(imgpath + a + '.jpg')
            rotated_img = rotate_image(img,angle)
            cv2.imwrite(rotated_imgpath + a + '_'+ str(angle) +'d.jpg',rotated_img)
            print (str(i) + ' has been rotated for '+ str(angle)+'°')
            tree = ET.parse(xmlpath + a + '.xml')
            root = tree.getroot()
            for box in root.iter('bndbox'):
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)
                x, y, w, h = rotate_xml(img, xmin, ymin, xmax, ymax, angle)
                cv2.rectangle(rotated_img, (x, y), (x+w, y+h), [0, 0, 255], 2)   #可在该步骤测试新画的框位置是否正确
    
    
                box.find('xmin').text = str(x)
                box.find('ymin').text = str(y)
                box.find('xmax').text = str(x+w)
                box.find('ymax').text = str(y+h)
            tree.write(rotated_xmlpath + a + '_'+ str(angle) +'d.xml')
    
            cv2.imwrite(rotated_imgpath + a + '_' + str(angle) + 'd.jpg', rotated_img)
    
            print (str(a) + '.xml has been rotated for '+ str(angle)+'°')