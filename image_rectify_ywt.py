import os
import glob
import cv2
import argparse
import numpy as np
import quaternion

ext = '.png'

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params_path', type=str, default='/data/endo_data/3D_endo/0326-采图/rgb/mei_out')
    
    parser.add_argument('--data_path', type=str, default='/data/endo_data/3D_endo/0326-采图/rgb')
    parser.add_argument('--rect_flag', type=str, default='longlati', choices=['perspective', 'longlati'])
    args = parser.parse_args()

    intri_left_yaml = os.path.join(args.params_path, 'camera_left.yaml')
    intri_right_yaml = os.path.join(args.params_path, 'camera_right.yaml')
    extrinsics_yaml = os.path.join(args.params_path, 'extrinsics.yaml')
    K_left, D_left, xi_left, width, height = read_intrinsics(intri_left_yaml)
    K_right, D_right, xi_right, _, _ = read_intrinsics(intri_right_yaml)
    R, T = read_extrinsics(extrinsics_yaml)

    print('width: ', width)
    print('height: ', height)
    print('K_left: ', K_left)
    print('D_left: ', D_left)
    print('xi_left: ', xi_left)
    print('K_right: ', K_right)
    print('D_right: ', D_right)
    print('xi_right: ', xi_right)
    print("R: ", R)
    print("T: ", T)
    ARC_TO_DEG =57.29577951308238
    degs = cv2.Rodrigues(R)[0].reshape(-1)*ARC_TO_DEG
    print(f'旋转角度： x: {degs[0]:.2f}, y: {degs[1]:.2f}, z:{degs[2]:.2f}')
    # exit()

    R_left, R_right = cv2.omnidir.stereoRectify(R, T)
    degs = cv2.Rodrigues(R_left)[0].reshape(-1)*ARC_TO_DEG
    print(f'旋转角度： x: {degs[0]:.2f}, y: {degs[1]:.2f}, z:{degs[2]:.2f}')
    degs = cv2.Rodrigues(R_right)[0].reshape(-1)*ARC_TO_DEG
    print(f'旋转角度： x: {degs[0]:.2f}, y: {degs[1]:.2f}, z:{degs[2]:.2f}')
    # print("R_left: ", R_left)
    # print("R_right: ", R_right)
    fn_left_image = sorted(glob.glob(os.path.join(args.data_path, f'left/*{ext}')))
    fn_right_image = sorted(glob.glob(os.path.join(args.data_path, f'right/*{ext}')))
    assert len(fn_left_image) == len(fn_right_image), "# ERROR: Left and right images must be equal"

    new_width = width
    new_height = height
    
    if args.rect_flag == 'perspective':
        rect_flag = cv2.omnidir.RECTIFY_PERSPECTIVE
    else: # args.rect_flag == 'longlati':
        rect_flag = cv2.omnidir.RECTIFY_LONGLATI
        
    new_K = np.array([
        [new_width /2, 0, 750],
        [0, new_height /2, 750],
        [0, 0, 1]
        ])
    # np.invert()
    print("new_K: ", new_K)




    out_undis_left = os.path.join(args.data_path, f"undis_camodolcal_{rect_flag}/left")
    out_undis_right = os.path.join(args.data_path, f"undis_camodolcal_{rect_flag}/right")
    out_undis_hconcat = os.path.join(args.data_path, f"undis_camodolcal_{rect_flag}/hconcat")
    os.makedirs(out_undis_left, exist_ok=True)
    os.makedirs(out_undis_right, exist_ok=True)
    os.makedirs(out_undis_hconcat, exist_ok=True)
    
    left_map1,left_map2 =cv2.omnidir.initUndistortRectifyMap(K_left, D_left, xi_left, R_left, new_K, (width, height),  flags=rect_flag,m1type=cv2.CV_32FC1)
    right_map1,right_map2 =cv2.omnidir.initUndistortRectifyMap(K_right, D_right, xi_right, R_right, new_K, (width, height),  flags=rect_flag,m1type=cv2.CV_32FC1)
    for i in range(len(fn_left_image)):
        print(f"Processing {fn_left_image[i]}")
        left_image = cv2.imread(fn_left_image[i])
        right_image = cv2.imread(fn_right_image[i])

        # undis_left = cv2.omnidir.undistortImage(left_image, K_left, D_left, xi_left, flags=rect_flag, Knew=new_K, new_size=(width, height), R=R_left)
        # undis_right = cv2.omnidir.undistortImage(right_image, K_right, D_right, xi_right, flags=rect_flag, Knew=new_K, new_size=(width, height), R=R_right)
        undis_left = cv2.remap(left_image,map1=left_map1,map2=left_map2,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        undis_right = cv2.remap(right_image,map1=right_map1,map2=right_map2,interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
        undis_hconcat = np.hstack([undis_left, undis_right])
        # print(undis_hconcat.shape)
  
        for j in range(50):
            cv2.line(undis_hconcat, pt1=(0, 54 * j), pt2=(undis_hconcat.shape[1] - 1, 54 * j), color=(0, 0, 255), thickness=2)

        cv2.imwrite(os.path.join(out_undis_left, f'{i:04d}.png'), undis_left)
        cv2.imwrite(os.path.join(out_undis_right, f'{i:04d}.png'), undis_right)
        cv2.imwrite(os.path.join(out_undis_hconcat, f'{i:04d}.png'), undis_hconcat)


if __name__=="__main__":
    main()
