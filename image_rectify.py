import os
import glob
import argparse
from pathlib import Path
import time
import shutil
import numpy as np
import quaternion
import cv2
from conner_h_rms import get_coners

ext = '.bmp'

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


def stereoRectify(R, T):
    assert (R.shape == (3, 3) or R.size == 3) and (R.dtype == np.float32 or R.dtype == np.float64)
    assert T.size == 3 and (T.dtype == np.float32 or T.dtype == np.float64)

    if R.shape == (3, 3):
        _R = R.astype(np.float64)
    elif R.size == 3:
        _R, _ = cv2.Rodrigues(R)
        _R = _R.astype(np.float64)

    _T = T.reshape(1, 3).astype(np.float64)

    R1 = np.zeros((3, 3), dtype=np.float64)
    R2 = np.zeros((3, 3), dtype=np.float64)

    R21 = _R.T

    T21 = R21 @ _T.T


    T21_norm = np.linalg.norm(T21)
    e1 = T21.T / T21_norm
    T21 =T21.reshape(-1)
    e2 = np.array([-T21[1], T21[0], 0.0])

    e2_norm = np.linalg.norm(e2)
    e2 = e2 / e2_norm
    e3 = np.cross(e1, e2)
    e3_norm = np.linalg.norm(e3)
    e3 = e3 / e3_norm

    R1[0] = e1
    R1[1] = e2
    R1[2] = e3

    R2 = R1 @ R21

    return R1, R2
def stereo_rectify(R,T):
    ARC_TO_DEG =57.29577951308238

    degs = cv2.Rodrigues(R)[0]
    degs_left = degs / 2
    R_left = cv2.Rodrigues(degs_left)[0]
    R_right = R.T @ R_left

    t = R_left @ T
    print('t: \n', t)
    print(f'{np.arctan(t[2]/t[0])*ARC_TO_DEG:.2f} {np.arctan(t[1]/t[0])*ARC_TO_DEG:.2f} ')

    e1 = t/ np.linalg.norm(t)
    e2 = np.array([-t[1], t[0], 0.0])
    e2 = e2 / np.linalg.norm(e2)
    
    e3 = np.cross(e1,e2)
    e3 = e3/ np.linalg.norm(e3)
    Rw = np.array([e1,e2,e3])
    
    R_left =Rw@ R_left
    R_right = Rw @ R_right
    
    return R_left, R_right

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


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--params_path', type=str, default='/Users/citianyu/Desktop/project/stereo_rectify/data/0521-采图-30度/mei_out')
    parser.add_argument('--data_path', type=str, default='/Users/citianyu/Desktop/project/stereo_rectify/data/0510-采图-0度/')
    # parser.add_argument('--data_path', type=str, default='/Users/citianyu/Desktop/project/stereo_rectify/data/0417-采图-15度/')
    # parser.add_argument('--data_path', type=str, default='/Users/citianyu/Desktop/project/stereo_rectify/data/0521-采图-30度/')
    parser.add_argument('--rect_flag', type=str, default='perspective', choices=['perspective', 'longlati'])
    args = parser.parse_args()
    data_path = args.data_path
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

    ARC_TO_DEG =57.29577951308238
    degs = cv2.Rodrigues(R)[0].reshape(-1)*ARC_TO_DEG
    print(f'旋转角度:       R: x: {degs[0]:.2f}, y: {degs[1]:.2f}, z:{degs[2]:.2f}')

    theta_a = 'ori'
    theta_a = 'div2'
    if theta_a == 'ori':
        R_left, R_right = stereoRectify(R, T)
    else:

        R_left, R_right = stereo_rectify(R,T)
        # print(f'旋转角度:  Rw: x: {degs[0]:.2f}, y: {degs[1]:.2f}, z:{degs[2]:.2f}')


    degs = cv2.Rodrigues(R_left)[0].reshape(-1)*ARC_TO_DEG

    print(f'旋转角度:  R_left: x: {degs[0]:.2f}, y: {degs[1]:.2f}, z:{degs[2]:.2f}')
    degs = cv2.Rodrigues(R_right)[0].reshape(-1)*ARC_TO_DEG
    print(f'旋转角度: R_right: x: {degs[0]:.2f}, y: {degs[1]:.2f}, z:{degs[2]:.2f}')
    # exit()
    fn_left_image = sorted(glob.glob(os.path.join(args.data_path, f'left/*{ext}')))
    fn_right_image = sorted(glob.glob(os.path.join(args.data_path, f'right/*{ext}')))
    # assert len(fn_left_image) == len(fn_right_image), "# ERROR: Left and right images must be equal"
    l_name, r_name = [], []
    for fn in fn_left_image:
        l_name.append(Path(fn).stem)
    for fn in fn_right_image:
        r_name.append(Path(fn).stem)
    fn_left_image, fn_right_image = [], []
    for fn in l_name:
        if fn not in r_name:
            continue
        fn_left_image.append(os.path.join(args.data_path, f'left/{fn}{ext}'))
        fn_right_image.append(os.path.join(args.data_path, f'right/{fn}{ext}'))
    new_width = width
    new_height = height
    new_width = width
    new_height = height
    
    if args.rect_flag == 'perspective':
        rect_flag = cv2.omnidir.RECTIFY_PERSPECTIVE
    else: # args.rect_flag == 'longlati':
        rect_flag = cv2.omnidir.RECTIFY_LONGLATI
        
    ratio = 2.8
    new_K = np.array([
        [new_width /ratio, 0, new_width/2],
        [0, new_height /ratio, new_height/2],
        [0, 0, 1]
        ])
    print(new_K)
    
    # left_map1,left_map2 =cv2.omnidir.initUndistortRectifyMap(K_left, D_left, xi_left, R_left, new_K, (new_width, new_height),  flags=rect_flag,m1type=cv2.CV_32FC1)
    # right_map1,right_map2 =cv2.omnidir.initUndistortRectifyMap(K_right, D_right, xi_right, R_right, new_K, (new_width, new_height),  flags=rect_flag,m1type=cv2.CV_32FC1)
    
    
    outp = os.path.join(args.data_path, f"undis_camodolcal_{ratio}")
    # if os.path.exists(outp):
    #     shutil.rmtree(outp)

    out_undis_left = os.path.join(args.data_path, f"undis_camodolcal_{ratio}/left_{theta_a}")
    out_undis_right = os.path.join(args.data_path, f"undis_camodolcal_{ratio}/right_{theta_a}")
    out_undis_hconcat = os.path.join(args.data_path, f"undis_camodolcal_{ratio}/hconcat_{theta_a}")
    # out_undis_lbl = os.path.join(args.data_path, f"undis_camodolcal_{ratio}/lbl")
    os.makedirs(out_undis_left, exist_ok=True)
    # os.makedirs(out_undis_lbl, exist_ok=True)
    os.makedirs(out_undis_right, exist_ok=True)
    os.makedirs(out_undis_hconcat, exist_ok=True)
    t0 = time.perf_counter()
    left_map1,left_map2 =cv2.omnidir.initUndistortRectifyMap(K_left, D_left, xi_left, R_left, new_K, (new_width, new_height),  flags=rect_flag,m1type=cv2.CV_32FC1)
    right_map1,right_map2 =cv2.omnidir.initUndistortRectifyMap(K_right, D_right, xi_right, R_right, new_K, (new_width, new_height),  flags=rect_flag,m1type=cv2.CV_32FC1)

    t1 = time.perf_counter()
    # print(t1-t0,len(fn_left_image))

    left_image_arr = [cv2.imread(fn_left_image[i]) for i in range(len(fn_left_image))]
    right_image_arr = [cv2.imread(fn_right_image[i]) for i in range(len(fn_left_image))]
    
    t0 = time.perf_counter()
    for left_image, right_image, left_images_path in zip(left_image_arr, right_image_arr,fn_left_image):

        name = Path(left_images_path).stem

        # undis_left = cv2.omnidir.undistortImage(left_image, K_left, D_left, xi_left, flags=rect_flag, Knew=new_K, new_size=(width, height), R=R_left)
        # undis_right = cv2.omnidir.undistortImage(right_image, K_right, D_right, xi_right, flags=rect_flag, Knew=new_K, new_size=(width, height), R=R_right)
        undis_left = cv2.remap(left_image,map1=left_map1,map2=left_map2,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        undis_right = cv2.remap(right_image,map1=right_map1,map2=right_map2,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        undis_hconcat = np.hstack([undis_left, undis_right])
  
        for j in range(50):
            cv2.line(undis_hconcat, pt1=(0, 54 * j), pt2=(undis_hconcat.shape[1] - 1, 54 * j), color=(0, 0, 255), thickness=1)

        cv2.imwrite(os.path.join(out_undis_left, f'{name}.png'), undis_left)
        cv2.imwrite(os.path.join(out_undis_right, f'{name}.png'), undis_right)
        cv2.imwrite(os.path.join(out_undis_hconcat, f'{name}.png'), undis_hconcat)
    t1 = time.perf_counter()

    print(out_undis_right)
    left_imgpoints, right_imgpoints, n_conners, names = get_coners(Path(out_undis_left),Path(out_undis_right))
    print(np.array(left_imgpoints).shape,np.array(right_imgpoints).shape)
    rmse = np.sqrt(((np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1])**2).mean())
    # camodolcal_3_x_no_subpix = np.sqrt(((np.array(left_imgpoints)[:,:,0,0]-np.array(right_imgpoints)[:,:,0,0])**2).mean())
    mae = np.abs(np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1]).mean()
    print(f'undis theta={theta_a} n={n_conners} RMSE : {rmse:.4f} MAE: {mae:.4f}')

    # return

    new_width = 1080*2
    new_height = 1080
    # ratio = 1.8
    new_K = np.array([
        [new_width /ratio, 0, new_width/2],
        [0, new_height /ratio, new_height/2],
        [0, 0, 1]
        ])
    print(new_K)
    
    left_map1,left_map2 =cv2.omnidir.initUndistortRectifyMap(K_left, D_left, xi_left, R_left, new_K, (new_width, new_height),  flags=rect_flag,m1type=cv2.CV_32FC1)
    right_map1,right_map2 =cv2.omnidir.initUndistortRectifyMap(K_right, D_right, xi_right, R_right, new_K, (new_width, new_height),  flags=rect_flag,m1type=cv2.CV_32FC1)

    t0 = time.perf_counter()
    for left_image, right_image, left_images_path in zip(left_image_arr, right_image_arr,fn_left_image):

        name = Path(left_images_path).stem

        undis_left = cv2.remap(left_image,map1=left_map1,map2=left_map2,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        undis_right = cv2.remap(right_image,map1=right_map1,map2=right_map2,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        # undis_hconcat = np.hstack([undis_left, undis_right])

        lbl = np.zeros((2160,3840,3),np.uint8)
        l = (3840-new_width)//2
        lbl[:new_height*2:2,l:l+new_width,:] = undis_left
        lbl[1:new_height*2:2,l:l+new_width,:] = undis_right

        cv2.putText(lbl, f'{ratio} {theta_a}', (150,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)

        p = Path(args.data_path) /  f"undis_camodolcal_{ratio}/lbl_{theta_a}/{name}.png"
        p.parent.mkdir(exist_ok=True,parents=True)
        cv2.imwrite(p.as_posix(),lbl)


if __name__=="__main__":
    np.set_printoptions(precision=5, suppress=True)
    main()


# stereoRectify
# 0510-采图-0度
# 旋转角度:       R: x: 0.29, y: 0.97, z:-1.56
# 旋转角度:  R_left: x: 0.01, y: -6.26, z:-0.25
# 旋转角度: R_right: x: -0.37, y: -7.22, z:1.30
# undis theta=ori n=63 RMSE : 0.5146 MAE: 0.3371

# 旋转角度:       R: x: 0.29, y: 0.97, z:-1.56
# 旋转角度:  Rw: x: -0.03, y: -6.74, z:0.52
# 旋转角度:  R_left: x: 0.16, y: -6.26, z:-0.25
# 旋转角度: R_right: x: -0.22, y: -7.22, z:1.29
# undis theta=div2 n=63 RMSE : 0.5175 MAE: 0.3374


# 0417-采图-15度
# 旋转角度:       R: x: 1.23, y: -12.28, z:-2.21
# 旋转角度:  R_left: x: -0.08, y: -10.13, z:0.90
# 旋转角度: R_right: x: -1.59, y: 2.15, z:2.98
# undis theta=ori n=44 RMSE : 0.7237 MAE: 0.4772

# 旋转角度:       R: x: 1.23, y: -12.28, z:-2.21
# 旋转角度:  Rw: x: -0.07, y: -4.00, z:1.91
# 旋转角度:  R_left: x: 0.69, y: -10.13, z:0.83
# 旋转角度: R_right: x: -0.82, y: 2.13, z:2.99
# undis theta=div2 n=44 RMSE : 0.6625 MAE: 0.4652

# 0521-采图-30度

# 旋转角度:       R: x: 1.24, y: 22.76, z:-0.89
# 旋转角度:  R_left: x: -0.51, y: -0.02, z:176.17
# 旋转角度: R_right: x: 34.37, y: -3.09, z:173.74
# undis theta=ori n=32 RMSE : 10.0108 MAE: 2.6528

# 旋转角度:       R: x: 1.24, y: 22.76, z:-0.89
# 旋转角度:  Rw: x: 16.89, y: 0.52, z:175.70
# 旋转角度:  R_left: x: -0.58, y: 2.06, z:176.16
# 旋转角度: R_right: x: 34.27, y: -1.04, z:173.52
# undis theta=div2 n=34 RMSE : 9.7192 MAE: 2.5394
