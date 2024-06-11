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

ARC_TO_DEG =57.29577951308238

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
# def rotation_matrix_to_vector(R):
#     tr = R[0,0]+R[1,1]+R[2,2]

#     c = (tr - 1) / 2
#     theta = np.arccos(np.clip(c, -1.0, 1.0))

#     u = np.array([(R[2, 1] - R[1, 2]) / (2 * np.sin(theta)),
#                 (R[0, 2] - R[2, 0]) / (2 * np.sin(theta)),
#                 (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))])

#     r = theta * u
#     return r
# 罗德里格斯公式的函数，用于旋转向量到旋转矩阵的转换
# def rodrigues(r):
#     theta = np.linalg.norm(r)
#     if theta == 0:
#         return np.eye(3)
#     else:
#         r = r / theta
#         K = np.array([[0, -r[2], r[1]],
#                     [r[2], 0, -r[0]],
#                     [-r[1], r[0], 0]])
#         return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
# def stereoRectify(R, T):
#     assert (R.shape == (3, 3) or R.size == 3) and (R.dtype == np.float32 or R.dtype == np.float64)
#     assert T.size == 3 and (T.dtype == np.float32 or T.dtype == np.float64)

#     if R.shape == (3, 3):
#         _R = R.astype(np.float64)
#     elif R.size == 3:
#         _R, _ = cv2.Rodrigues(R)
#         _R = _R.astype(np.float64)

#     _T = T.reshape(1, 3).astype(np.float64)

#     R1 = np.zeros((3, 3), dtype=np.float64)
#     R2 = np.zeros((3, 3), dtype=np.float64)
#     R21 = _R.T
#     T21 = -R21 @ _T.T

#     T21_norm = np.linalg.norm(T21)
#     e1 = T21.T / T21_norm
#     T21 =T21.reshape(-1)
#     e2 = np.array([-T21[1], T21[0], 0.0])

#     e2_norm = np.linalg.norm(e2)
#     e2 = e2 / e2_norm
#     e3 = np.cross(e1, e2)
#     e3_norm = np.linalg.norm(e3)
#     e3 = e3 / e3_norm

#     R1[0] = e1
#     R1[1] = e2
#     R1[2] = e3

#     R2 = R1 @ R21

#     return R1, R2

def stereo_rectify(R,T):
    ARC_TO_DEG =57.29577951308238

    degs = cv2.Rodrigues(R)[0]
    degs_left = degs / 2
    R_left = cv2.Rodrigues(degs_left)[0]
    R_right = R.T @ R_left

    t = R_left @ T

    e1 = t/ np.linalg.norm(t)
    e2 = np.array([-t[1], t[0], 0.0])
    e2 = e2 / np.linalg.norm(e2)
    
    e3 = np.cross(e1,e2)
    e3 = e3/ np.linalg.norm(e3)
    Rw = np.array([e1,e2,e3])
    
    R_left =Rw@ R_left
    R_right = Rw @ R_right
    
    return R_left, R_right

def morph_close(img,k_size=5):
    kernel = np.ones((k_size,k_size), np.uint8)

# 进行开运算
    dilation = cv2.dilate(img, kernel, iterations=1)
    return dilation
def objectpoint2_lrpoint(objectPoints_ori,R_object2left, R, T):
    objectPoints_l= objectPoints_ori.reshape((-1,3)).transpose(1,0)
    objectPoints_l = R_object2left @ objectPoints_l

    objectPoints_l = objectPoints_l.transpose(1,0)

    objectPoints_r= objectPoints_ori.reshape((-1,3)).transpose(1,0)
    objectPoints_r = R_object2left@objectPoints_r -T[:,None]
    objectPoints_r = R @ objectPoints_r

    objectPoints_r = objectPoints_r.transpose(1,0)
    return objectPoints_l, objectPoints_r
        

def find_left_view_point(R_object2left,depth, base_T, h, R,T,K_left,xi_left,D_left,K_right,xi_right,D_right):
    l_max = -depth/ np.tan(20/180*np.pi)*1.5
    r_vec = cv2.Rodrigues(R.T)[0]
    while l_max < 0:
        opoint = np.array([l_max+base_T[0]/2, h, depth]).reshape((3,1))
        op_l, op_r = objectpoint2_lrpoint(opoint, R_object2left, R, T)

        imgp_left_l, _ = cv2.omnidir.projectPoints(op_l.reshape((1,1,3)),np.zeros(3),np.zeros(3),K_left,xi_left[0,0],D_left)
        imgp_right_l, _ = cv2.omnidir.projectPoints(op_r.reshape((1,1,3)),r_vec,-T,K_right,xi_right[0,0],D_right)
        imgp_left_l = np.round(imgp_left_l).astype('int32').reshape(-1)
        imgp_right_l = np.round(imgp_right_l).astype('int32').reshape(-1)
        if imgp_left_l[0]>0 and imgp_right_l[0]>0:
            break
        l_max += 0.1
    
    r_max = depth/ np.tan(20/180*np.pi)*1.5
    while r_max > 0:
        opoint = np.array([r_max+base_T[0]/2, h, depth]).reshape((3,1))
        op_l, op_r = objectpoint2_lrpoint(opoint, R_object2left, R, T)

        imgp_left_r, _ = cv2.omnidir.projectPoints(op_l.reshape((1,1,3)),np.zeros(3),np.zeros(3),K_left,xi_left[0,0],D_left)
        imgp_right_r, _ = cv2.omnidir.projectPoints(op_r.reshape((1,1,3)),r_vec,-T,K_right,xi_right[0,0],D_right)
        imgp_left_r = np.round(imgp_left_r).astype('int32').reshape(-1)
        imgp_right_r = np.round(imgp_right_r).astype('int32').reshape(-1)
        if imgp_left_r[0]<1500 and imgp_right_r[0]<1500:
            break
        r_max -= 0.1

    # print(imgp_left, imgp_right)
    return (l_max, imgp_left_l, imgp_right_l), (r_max, imgp_left_r, imgp_right_r)

def main():
    # data_path = '/Users/citianyu/Desktop/project/stereo_rectify/data/0521-采图-30度/'
    data_path = '/Users/citianyu/Desktop/project/stereo_rectify/data/0510-采图-0度/'
    # data_path = '/Users/citianyu/Desktop/project/stereo_rectify/data/0417-采图-15度/'

    # data_path = '/Users/citianyu/Desktop/project/stereo_rectify/data/0521-采图-30度/'
    cv2.namedWindow("WindowName", cv2.WINDOW_FULLSCREEN)
    new_width = 2160
    new_height = 980
    cv2.resizeWindow("WindowName", new_width+400, new_height*2)
    # cv2.moveWindow("WindowName", (3840-new_width+400)//2, 0,new_width+400, new_height*2)
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
    rect_flag=cv2.omnidir.RECTIFY_LONGLATI
    n = 0 
    base_depth = 10
    grid_h, grid_w = 51,51
    objectColor = np.random.randint(0,255,size=(grid_h,grid_w,3)).astype('uint8')
    # objectColor[:] = 255
    objectColor = cv2.GaussianBlur(objectColor,(5,5),0)
    objectColor = cv2.GaussianBlur(objectColor,(5,5),0)
    objectColor = cv2.GaussianBlur(objectColor,(5,5),0)
    base_deg = 0
    base_T = np.array([2.5,0,0])
    virtual_depth = 20
    dem = np.zeros((grid_h,grid_w))
    dem[np.random.randint(0,grid_h),np.random.randint(0,grid_w)] = 5
    dem[np.random.randint(0,grid_h),np.random.randint(0,grid_w)] = -5
    dem[np.random.randint(0,grid_h),np.random.randint(0,grid_w)] = 5
    dem[np.random.randint(0,grid_h),np.random.randint(0,grid_w)] = -5
    dem[np.random.randint(0,grid_h),np.random.randint(0,grid_w)] = 5
    dem[np.random.randint(0,grid_h),np.random.randint(0,grid_w)] = -5
    dem[np.random.randint(0,grid_h),np.random.randint(0,grid_w)] = 5
    dem[np.random.randint(0,grid_h),np.random.randint(0,grid_w)] = -5
    dem = cv2.GaussianBlur(dem*2,(5,5),0)
    while True:
        degs = np.array([0,base_deg/ARC_TO_DEG*2,0])
        R = cv2.Rodrigues(degs)[0]
        
        R_object2left = cv2.Rodrigues(np.array([0,base_deg/ARC_TO_DEG,0]))[0].T


        T = R_object2left @ base_T
        print("R: \n", R)
        degs = cv2.Rodrigues(R)[0].reshape(-1)*ARC_TO_DEG
        print(f'旋转角度:  R: x: {degs[0]:.2f}, y: {degs[1]:.2f}, z:{degs[2]:.2f}')
        print("T: \n", T, )

        R_left, R_right = stereo_rectify(R,T)
        degs = cv2.Rodrigues(R_left)[0].reshape(-1)*ARC_TO_DEG

        print(f'旋转角度:  R_left: x: {degs[0]:.2f}, y: {degs[1]:.2f}, z:{degs[2]:.2f}')
        degs = cv2.Rodrigues(R_right)[0].reshape(-1)*ARC_TO_DEG
        print(f'旋转角度: R_right: x: {degs[0]:.2f}, y: {degs[1]:.2f}, z:{degs[2]:.2f}')
        objectPoints_ori = np.zeros([grid_h,grid_w,3],np.float64)
        depth = base_depth


        s = 0.4
        n += 1
        for i in range(grid_h):
            for j in range(grid_w):
                objectPoints_ori[i,j,0]=i-grid_h//2
                objectPoints_ori[i,j,1]=j-grid_w//2
                objectPoints_ori[i,j,2]=depth  +  1* dem[i,j]


        objectPoints_ori[...,[0,1]] =objectPoints_ori[...,[0,1]]*s
        objectPoints_ori[...,0] += base_T[0]/2
        
        objectPoints_l, objectPoints_r = objectpoint2_lrpoint(objectPoints_ori,R_object2left, R, T)
        
        # objectPoints_l= objectPoints_ori.reshape((-1,3)).transpose(1,0)
        # objectPoints_l = R_object2left @ objectPoints_l
        # print(objectPoints_l.reshape((3,21,21))[:,10,10],'L')
        # objectPoints_l = objectPoints_l.transpose(1,0).reshape((-1,1,3))
        objectColor_l = objectColor.reshape((-1,3))
        
        l_max, r_max = find_left_view_point(R_object2left,base_depth,base_T, 0, R,T,K_left,xi_left,D_left,K_right,xi_right,D_right)
        
        imgp = cv2.omnidir.projectPoints(objectPoints_l.reshape((-1,1,3)),np.zeros(3),np.zeros(3),K_left,xi_left[0,0],D_left)
        idxs = imgp[0][:,0,:]
        img_left = np.zeros([height,width,3],np.uint8)
        idxs = np.round(idxs).astype('int32')
        objectColor_l=objectColor_l[idxs[:,0]>=0,:]
        idxs = idxs[idxs[:,0]>=0,:]
        objectColor_l=objectColor_l[idxs[:,0]<width,:]
        idxs = idxs[idxs[:,0]<width,:]
        objectColor_l=objectColor_l[idxs[:,1]>=0,:]
        idxs = idxs[idxs[:,1]>=0,:]
        objectColor_l=objectColor_l[idxs[:,1]<height,:]
        idxs = idxs[idxs[:,1]<height,:]
        # img_left[:] = 255
        img_left[idxs[:,1],idxs[:,0],:]=objectColor_l
        img_left = morph_close(img_left)
        # img_left[img_left==0]=255
        # print((l_max[1][0,0,0],l_max[1][0,0,1]))
        cv2.circle(img_left, (l_max[1][0],l_max[1][1]), 10, (0,0,255),-1)
        cv2.circle(img_left, (r_max[1][0],r_max[1][1]), 10, (0,255,0),-1)

        
        objectColor_r = objectColor.reshape((-1,3))
        imgp = cv2.omnidir.projectPoints(objectPoints_r.reshape((-1,1,3)),np.zeros(3),np.zeros(3),K_right,xi_right[0,0],D_right)
        idxs = imgp[0][:,0,:]
        objectColor_r = objectColor.reshape((-1,3))
        
        img_right = np.zeros([height,width,3],np.uint8)
        
        idxs = np.round(idxs).astype('int32')
        objectColor_r=objectColor_r[idxs[:,0]>=0,:]
        idxs = idxs[idxs[:,0]>=0,:]
        objectColor_r=objectColor_r[idxs[:,0]<width,:]
        idxs = idxs[idxs[:,0]<width,:]
        objectColor_r=objectColor_r[idxs[:,1]>=0,:]
        idxs = idxs[idxs[:,1]>=0,:]
        objectColor_r=objectColor_r[idxs[:,1]<height,:]
        idxs = idxs[idxs[:,1]<height,:]
        # img_right[:] = 255
        img_right[idxs[:,1],idxs[:,0],:]=objectColor_r
        img_right = morph_close(img_right)
        # img_right[img_right==0]=255
        cv2.circle(img_right, (l_max[2][0],l_max[2][1]), 10, (0,0,255),-1)
        cv2.circle(img_right, (r_max[2][0],r_max[2][1]), 10, (0,255,0),-1)
        
        ratio = 2.0
        new_K = np.array([
            [new_width /ratio, 0, new_width/2],
            [0, new_height /ratio, new_height/2],
            [0, 0, 1]
            ])

        
        left_map1,left_map2 =cv2.omnidir.initUndistortRectifyMap(K_left, D_left, xi_left, R_left, new_K, (new_width, new_height),  flags=rect_flag,m1type=cv2.CV_32FC1)
        right_map1,right_map2 =cv2.omnidir.initUndistortRectifyMap(K_right, D_right, xi_right, R_right, new_K, (new_width, new_height),  flags=rect_flag,m1type=cv2.CV_32FC1)
        undis_left = cv2.remap(img_left,map1=left_map1,map2=left_map2,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        undis_right = cv2.remap(img_right,map1=right_map1,map2=right_map2,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
            
        global_shift = int(np.linalg.norm(T) * new_K[0,0] / virtual_depth)
        if global_shift >=400:
            global_shift = 400
        print(global_shift)

        lbl = np.zeros((new_height*2, new_width+400,3),np.uint8)
        lbl[::2,200-global_shift//2:200-global_shift//2+new_width,:] = undis_left
        lbl[1::2,200+global_shift//2:200+global_shift//2+new_width,:] = undis_right


        cv2.putText(lbl, f'base_depth {base_depth} depth: {objectPoints_ori[...,2].min():.1f}-{objectPoints_ori[...,2].max():.1f}', (150,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(lbl, f'grid {s:.2f}', (150,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(lbl, f'deg  {base_deg*2:.2f}', (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
        
        cv2.putText(lbl, f'range  {l_max[0]:.2f} - {r_max[0]:.2f}', (150,250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(lbl, f'virtual_depth  {virtual_depth}', (150,300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(lbl, f'global_shift  {global_shift}', (150,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
        
        cv2.circle(lbl,(new_width//2, new_height),1,(255,255,255),1)
        
        cv2.imwrite('output/img_left.png',img_left)
        cv2.imwrite('output/img_right.png',img_right)
        cv2.imwrite('output/undis_left.png',undis_left)
        cv2.imwrite('output/undis_right.png',undis_right)
        cv2.imwrite('output/lbl.png',lbl)
        
        cv2.imshow("WindowName",lbl)

        key = cv2.waitKey(50)
        print(key)
        if key == 0: # up  
            base_depth += 1
        if key == 1: # down 
            base_depth -= 1
        if key == 2: # left
            base_deg -= 0.5
        if key == 3:
            base_deg += 0.5
        if key & 0xFF == ord('q'):
            return
        if key & 0xFF == ord('q'):
            return
        if key & 0xFF == ord('w'):
            virtual_depth += 1
        if key & 0xFF == ord('s'):
            virtual_depth -= 1
            global_shift = int(np.linalg.norm(T) * new_K[0,0] / virtual_depth)
            if global_shift >=400:
                virtual_depth += 1
        # if key & 0xFF == ord('q'):
            

if __name__=='__main__':
    main()
    
    