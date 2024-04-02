from pathlib import Path
import re
import numpy as np
import os
import cv2
from tqdm import tqdm
from FisheyeCalibrate import FisheyeCalibrator
from OmnidirCalibrate import OmnidirCalibrator
from DepthEstimation import DepthEstimator
from PinholeCalibrate import PinholeCalibrator
def rgb2rggb(rgb):
    R = rgb[::2, ::2, 0:1]
    Gr = rgb[::2, 1::2, 1:2]
    Gb = rgb[1::2, ::2, 1:2]
    B = rgb[1::2, 1::2, 2:]
    rggb = np.concatenate((R, Gr, Gb, B), axis=2)
    return rggb


def rggb2bayer(x):
    h, w = x.shape[:2]
    return x.reshape(h, w, 2, 2).transpose(0, 2, 1, 3).reshape(2*h, 2*w)


def bayer2rggb(x):
    h, w = x.shape[:2]
    return x.reshape(h//2, 2, w//2, 2).transpose(0, 2, 1, 3).reshape(h//2, w//2, 4)


def rggb2rgb(rggb):
    return cv2.cvtColor(np.uint16(rggb2bayer(rggb)*65535), cv2.COLOR_BAYER_BG2RGB)/65535.0

def decode_raw(img_path):
    raw = np.fromfile(img_path,dtype=np.uint16)
    h, w = 1080, 1352
    raw = raw[:h*w].reshape(h,w)

    raw = np.left_shift(raw, 6).astype(np.float32)/ 64.
    rggb = bayer2rggb(raw)
    rggb = (rggb / 1023).clip(0,1)
    # rgb = rggb2rgb(rggb)
    # cv2.imwrite('/data/temp/tmp.png',(rgb*255)[...,::-1].astype(np.uint8))
    return rggb
def _create_object_point(_board, _square_size):
    # _ -> object points grid refered board
    object_point = np.zeros((1, _board[0]*_board[1], 3), np.float32)
    object_point[0,:,:2] = np.mgrid[0:_board[0], 0:_board[1]].T.reshape(-1, 2)
    object_point = object_point*_square_size 
    return object_point

def opencv_save_img_pathlib(savepath, img):

    savepath.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(savepath.as_posix(), img)
    
if __name__=="__main__":
    datadir = Path('/data/endo_data/3D_endo/stereo_231222')
    # print(_create_object_point((8,11),0.0052))

    # datadir = Path('/data/endo_data/3D_endo/0315-采图-1352-1080/rgb')
    # datadir = Path('/data/endo_data/3D_endo/0318-采图/rgb')
    # datadir = Path('/data/endo_data/3D_endo/0326-采图/rgb')
    # datadir = Path('/data/endo_data/3D_endo/0327-采图/rgb')
    lefts, rights, images = [], [], []
    img_dic = {}
    shape = None
    right_imgpoints, left_imgpoints, object_points = [], [], []
    n = 0
    for left_path in tqdm(sorted(list(datadir.glob('left/*.jpg'))+list(datadir.glob('left/*.png')))):
        n += 1
        id = left_path.stem

        right_path = datadir /f'right/{left_path.name}'
        if not right_path.exists():
            continue

        left_rgb = cv2.imread(left_path.as_posix())/255.
        right_rgb = cv2.imread(right_path.as_posix())/255.
        lefts.append((left_rgb*255).astype(np.uint8))
        rights.append((right_rgb*255).astype(np.uint8))
        right_image = cv2.cvtColor((right_rgb*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        left_image = cv2.cvtColor((left_rgb*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        

        flag = cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE
        right_ret, right_corner = cv2.findChessboardCorners(right_image, (8,11), flag)
        left_ret, left_corner = cv2.findChessboardCorners(left_image, (8,11), flag)

        shape = left_rgb.shape[:2]
        gray = (np.concatenate([left_rgb, right_rgb], axis=1)*255).astype(np.uint8)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        if not left_ret  or not right_ret:
            continue
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        right_corner =cv2.cornerSubPix(right_image,right_corner,(3,3),(-1,-1),subpix_criteria)
        right_imgpoints.append(right_corner.reshape(1,-1,2))
        
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        cv2.cornerSubPix(left_image,left_corner,(3,3),(-1,-1),subpix_criteria)
        left_imgpoints.append(left_corner.reshape(1,-1,2))
        object_point=_create_object_point((8,11),0.0052)
        object_points.append(object_point)

        
        
        images.append(gray)
        
        left_chess = cv2.drawChessboardCorners((right_rgb*255).astype(np.uint8), (8,11), right_corner,left_ret)
        out_path= datadir/'chess'/'right'/ f'{id}.png'
        out_path.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(out_path.as_posix(), left_chess)
        right_chess = cv2.drawChessboardCorners((left_rgb*255).astype(np.uint8), (8,11), left_corner,right_ret)
        out_path= datadir/'chess'/'left'/ f'{id}.png'
        out_path.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(out_path.as_posix(), right_chess)
        

        img_dic[id]={
            'left': (left_rgb*255).astype(np.uint8),
            'right': (right_rgb*255).astype(np.uint8),
            'object_point': object_point,
            'left_corner': left_corner.reshape(1,-1,2),
            'right_corner': right_corner.reshape(1,-1,2)
        }

    # pinholeCalibrator = PinholeCalibrator(img_dic=img_dic,image_size=left_chess.shape[:2])
    # pinholeCalibrator.stereo_calibrate()
    # pinholeCalibrator.stereo_rectify()
    # pinholeCalibrator.create_rectify_map()
    # for id in img_dic.keys():
    #     img = img_dic[id]
    #     out_path = datadir/'pinhole'/'rec'/'hconcat'/ f'{id}.png'
    #     out_path.parent.mkdir(parents=True,exist_ok=True)
    #     rectified_image, left_rectified_image, right_rectified_image = pinholeCalibrator.get_rectified_image(
    #         left_image=img_dic[id]['left'],
    #         right_image=img_dic[id]['right'],
    #         save=out_path.as_posix()
    #         )
    #     opencv_save_img_pathlib(datadir/'pinhole'/'rec'/'left'/ f'{id}.png', left_rectified_image)
    #     opencv_save_img_pathlib(datadir/'pinhole'/'rec'/'right'/ f'{id}.png', right_rectified_image)
    
    
    # fisheyeCalibrator = FisheyeCalibrator(img_dic=img_dic,image_size=left_chess.shape[:2])
    # fisheyeCalibrator.stereo_calibrate()
    # fisheyeCalibrator.stereo_rectify()
    # fisheyeCalibrator.create_rectify_map()
    # fisheyeCalibrator.create_equirectangular_map()
    # for id in img_dic.keys():
    #     img = img_dic[id]
    #     out_path = datadir/'fisheye'/'rec'/'hconcat'/ f'{id}.png'
    #     out_path.parent.mkdir(parents=True,exist_ok=True)
    #     rectified_image, left_rectified_image, right_rectified_image = fisheyeCalibrator.get_rectified_image(
    #         left_image=img_dic[id]['left'],
    #         right_image=img_dic[id]['right'],
    #         save=out_path.as_posix()
    #         )
    #     opencv_save_img_pathlib(datadir/'fisheye'/'rec'/'left'/ f'{id}.png', left_rectified_image)
    #     opencv_save_img_pathlib(datadir/'fisheye'/'rec'/'right'/ f'{id}.png', right_rectified_image)
        
    #     out_path = datadir/'fisheye'/'erec'/'hconcat'/ f'{id}.png'
    #     out_path.parent.mkdir(parents=True,exist_ok=True)
    #     eqrec_image,left_eqrec_image,right_eqrec_image = fisheyeCalibrator.get_equirectangular_image(
    #             left_image=left_rectified_image,
    #             right_image=right_rectified_image,
    #             save=out_path.as_posix()
    #             )
    #     opencv_save_img_pathlib(datadir/'fisheye'/'erec'/'left'/ f'{id}.png', left_eqrec_image)
    #     opencv_save_img_pathlib(datadir/'fisheye'/'erec'/'right'/ f'{id}.png', right_eqrec_image)



    omnidirCalibrator = OmnidirCalibrator(img_dic=img_dic,image_size=left_chess.shape[:2])
    omnidirCalibrator.stereo_calibrate()
    o = omnidirCalibrator


    for id in img_dic.keys():
        img = img_dic[id]
        # out_path = datadir/'omidir'/'rec'/ 'hconcat' /f'{id}.png'
        # out_path.parent.mkdir(parents=True,exist_ok=True)
        # out_left, out_right = omnidirCalibrator.rectify(
        #     image_left=img_dic[id]['left'],
        #     image_right=img_dic[id]['right'],
        #     save=out_path.as_posix()
        #     )
        out_path = datadir/'omidir'/'erec'/ 'hconcat'/f'{id}.png'
        out_path.parent.mkdir(parents=True,exist_ok=True)
        rectified_image, image_left_rec, image_right_rec = omnidirCalibrator.stereo_rectify(
            image_left=img_dic[id]['left'],
            image_right=img_dic[id]['right'],
            save=out_path.as_posix()
            )
        opencv_save_img_pathlib(datadir/'omidir'/'erec'/'left'/ f'{id}.png', image_left_rec)
        opencv_save_img_pathlib(datadir/'omidir'/'erec'/'right'/ f'{id}.png', image_right_rec)



    