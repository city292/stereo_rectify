import cv2
from pathlib import Path
import os
import sys
import hashlib
import tqdm
import numpy as np


from pathlib import Path
import time
import shutil

import quaternion


def read_intrinsics(intrinsics_yaml):
    fs = cv2.FileStorage(intrinsics_yaml, cv2.FILE_STORAGE_READ)

    width = int(fs.getNode('image_width').real())
    height = int(fs.getNode('image_height').real())

    _mirror_parameters = fs.getNode('mirror_parameters')
    _xi = _mirror_parameters.getNode('xi').real()

    _distortion_parameters = fs.getNode('distortion_parameters')
    _k1 = _distortion_parameters.getNode('k1').real()
    _k2 = _distortion_parameters.getNode('k2').real()
    _p1 = _distortion_parameters.getNode('p1').real()
    _p2 = _distortion_parameters.getNode('p2').real()

    _projection_parameters = fs.getNode('projection_parameters')
    _gamma1 = _projection_parameters.getNode('gamma1').real()
    _gamma2 = _projection_parameters.getNode('gamma2').real()
    _u0 = _projection_parameters.getNode('u0').real()
    _v0 = _projection_parameters.getNode('v0').real()

    fs.release()

    K = np.array([
        [_gamma1, 0, _u0],
        [0, _gamma2, _v0],
        [0, 0, 1]
        ])
    D = np.array([_k1, _k2, _p1, _p2])
    xi = np.array([[_xi]])

    return K, D, xi, width, height

def read_extrinsics(extrinsics_yaml):

    fs = cv2.FileStorage(extrinsics_yaml, cv2.FILE_STORAGE_READ)

    transform = fs.getNode("transform")
    _q_x = transform.getNode("q_x").real()
    _q_y = transform.getNode("q_y").real()
    _q_z = transform.getNode("q_z").real()
    _q_w = transform.getNode("q_w").real()
    _t_x = transform.getNode("t_x").real()
    _t_y = transform.getNode("t_y").real()
    _t_z = transform.getNode("t_z").real()

    fs.release()

    _q = np.quaternion(_q_w, _q_x, _q_y, _q_z)
    R = quaternion.as_rotation_matrix(_q)

    T = np.array([_t_x, _t_y, _t_z])

    return R, T
# 旋转矩阵转换为旋转向量的函数
def rotation_matrix_to_vector(R):
    tr = R[0,0]+R[1,1]+R[2,2]

    c = (tr - 1) / 2
    theta = np.arccos(np.clip(c, -1.0, 1.0))

    u = np.array([(R[2, 1] - R[1, 2]) / (2 * np.sin(theta)),
                (R[0, 2] - R[2, 0]) / (2 * np.sin(theta)),
                (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))])

    r = theta * u
    return r
# 罗德里格斯公式的函数，用于旋转向量到旋转矩阵的转换
def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    else:
        r = r / theta
        K = np.array([[0, -r[2], r[1]],
                    [r[2], 0, -r[0]],
                    [-r[1], r[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def morph_close(img,k_size=5):
    kernel = np.ones((k_size,k_size), np.uint8)

# 进行开运算
    dilation = cv2.dilate(img, kernel, iterations=1)
    return dilation
def main():
    data_path = '/Users/citianyu/Desktop/project/stereo_rectify/data/0521-采图-30度/'
    data_path = '/Users/citianyu/Desktop/project/stereo_rectify/data/0510-采图-0度/'
    # data_path = '/Users/citianyu/Desktop/project/stereo_rectify/data/0417-采图-15度/'

    # data_path = '/Users/citianyu/Desktop/project/stereo_rectify/data/0521-采图-30度/'
    intri_left_yaml = (Path(data_path) / 'mei_out'/ 'camera_left.yaml').as_posix()
    intri_right_yaml = (Path(data_path) / 'mei_out'/  'camera_right.yaml').as_posix()
    extrinsics_yaml = (Path(data_path) / 'mei_out'/  'extrinsics.yaml').as_posix()
    K_left, D_left, xi_left, width, height = read_intrinsics(intri_left_yaml)
    K_right, D_right, xi_right, _, _ = read_intrinsics(intri_right_yaml)
    R, T = read_extrinsics(extrinsics_yaml)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format},suppress=False)
    print('width: ', width)
    print('height: ', height)
    print('K_left: \n', K_left)
    print('D_left: \n', D_left)
    print('xi_left: ', xi_left[0,0])
    print('K_right: \n', K_right)
    print('D_right: \n', D_right)
    print('xi_right: ', xi_right[0,0])
    print("R: \n", R)
    print("T: \n", T)
    objectPoints = np.zeros([20,20,3],np.float64)
    objectPoints[...,2]=20
    s = 0.2
    for i in range(20):
        objectPoints[i,:,0]=(i-10)*np.abs(i-10)
        objectPoints[:,i,1]=(i-10)*np.abs(i-10)
    objectPoints[...,[0,1]] =objectPoints[...,[0,1]]*s
    objectPoints= objectPoints.reshape((-1,1,3))
    print(objectPoints.dtype)
    imgp = cv2.omnidir.projectPoints(objectPoints,np.zeros(3),np.zeros(3),K_left,xi_left[0,0],D_left)
    idxs = imgp[0]
    img_left = np.zeros([1500,1500,3],np.uint8)

    img_left[np.round(idxs[:,0,0]).astype(np.int32),np.round(idxs[:,0,1]).astype(np.int32),:]=255
    cv2.imwrite('./output/img_left.png',morph_close(img_left))
    
    r_vec = rotation_matrix_to_vector(R)
    imgp = cv2.omnidir.projectPoints(objectPoints,r_vec,T,K_right,xi_right[0,0],D_right)
    idxs = imgp[0]
    img_right = np.zeros([1500,1500,3],np.uint8)

    img_right[np.round(idxs[:,0,0]).astype(np.int32),np.round(idxs[:,0,1]).astype(np.int32),:]=255
    cv2.imwrite('./output/img_right.png',morph_close(img_right))
if __name__=='__main__':
    main()
    
    