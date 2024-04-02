from pathlib import Path
import re
import numpy as np
import os
import cv2
from tqdm import tqdm


def get_coners(datadir):
    right_imgpoints, left_imgpoints, object_points = [], [], []
    n = 0
    for left_path in sorted(list(datadir.glob('left/*.png'))+list(datadir.glob('left/*.jpg'))):
        
        id = left_path.stem

        right_path = datadir /f'right/{left_path.name}'
        if not right_path.exists():
            continue

        left_rgb = cv2.imread(left_path.as_posix())/255.
        right_rgb = cv2.imread(right_path.as_posix())/255.

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
        n += 1
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 300, 0.01)
        right_corner =cv2.cornerSubPix(right_image,right_corner,(3,3),(-1,-1),subpix_criteria)

        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 300, 0.01)

        left_corner =cv2.cornerSubPix(left_image,left_corner,(3,3),(-1,-1),subpix_criteria)

        left_imgpoints.append(left_corner)
        right_imgpoints.append(right_corner)
        
        left_chess = cv2.drawChessboardCorners((right_rgb*255).astype(np.uint8), (8,11), right_corner,left_ret)
        out_path= datadir/'chess'/'right'/ f'{id}.png'
        out_path.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(out_path.as_posix(), left_chess)
        right_chess = cv2.drawChessboardCorners((left_rgb*255).astype(np.uint8), (8,11), left_corner,right_ret)
        out_path= datadir/'chess'/'left'/ f'{id}.png'
        out_path.parent.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(out_path.as_posix(), right_chess)
        
    return left_imgpoints, right_imgpoints, n
    
if __name__=="__main__":

    # left_imgpoints, right_imgpoints, nfish = get_coners(Path('/data/endo_data/3D_endo/stereo_231222/fisheye/rec'))
        

    # print(np.array(left_imgpoints).shape,np.array(right_imgpoints).shape)
    # coner_rm_fisheye = np.sqrt(((np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1])**2).mean())
    # print(coner_rm_fisheye)
    # coner_rm_fisheye_x = np.sqrt(((np.array(left_imgpoints)[:,:,0,0]-np.array(right_imgpoints)[:,:,0,0])**2).mean())
    # print(coner_rm_fisheye_x)
    # mae_fisheye = np.abs(np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1]).mean()
    
    


    # left_imgpoints, right_imgpoints, n_undis_1 = get_coners(Path('/data/endo_data/3D_endo/0327-采图/rgb/undis_camodolcal_1'))
    # print(np.array(left_imgpoints).shape,np.array(right_imgpoints).shape)
    # coner_rm_1 = np.sqrt(((np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1])**2).mean())
    # coner_rm_1_x = np.sqrt(((np.array(left_imgpoints)[:,:,0,0]-np.array(right_imgpoints)[:,:,0,0])**2).mean())
    # mae_unidis_1 = np.abs(np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1]).mean()
    

    # left_imgpoints, right_imgpoints, n_undis_3 = get_coners(Path('/data/endo_data/3D_endo/stereo_231222/undis_3'))

    # print(np.array(left_imgpoints).shape,np.array(right_imgpoints).shape)
    # coner_rm_3 = np.sqrt(((np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1])**2).mean())
    # coner_rm_3_x = np.sqrt(((np.array(left_imgpoints)[:,:,0,0]-np.array(right_imgpoints)[:,:,0,0])**2).mean())
    # mae_unidis_3 = np.abs(np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1]).mean()
    # left_imgpoints, right_imgpoints, n_pinhole = get_coners(Path('/data/endo_data/3D_endo/stereo_231222/pinhole/rec'))
   
    # print(np.array(left_imgpoints).shape,np.array(right_imgpoints).shape)
    # pinhole_rmse = np.sqrt(((np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1])**2).mean())
    # pinhole_x = np.sqrt(((np.array(left_imgpoints)[:,:,0,0]-np.array(right_imgpoints)[:,:,0,0])**2).mean())
    # mae_pinhole = np.abs(np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1]).mean()
    # left_imgpoints, right_imgpoints, n_omidir = get_coners(Path('/data/endo_data/3D_endo/stereo_231222/omidir/erec'))
   
   
    # print(np.array(left_imgpoints).shape,np.array(right_imgpoints).shape)
    # omidir_rmse = np.sqrt(((np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1])**2).mean())
    # omidir_x = np.sqrt(((np.array(left_imgpoints)[:,:,0,0]-np.array(right_imgpoints)[:,:,0,0])**2).mean())
    # mae_omidir = np.abs(np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1]).mean()


    left_imgpoints, right_imgpoints, n_camodolcal_3 = get_coners(Path('/data/endo_data/3D_endo/0326-采图/rgb/undis_camodolcal_3'))
   
    print(np.array(left_imgpoints).shape,np.array(right_imgpoints).shape)
    camodolcal_3_rmse = np.sqrt(((np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1])**2).mean())

    camodolcal_3_x = np.sqrt(((np.array(left_imgpoints)[:,:,0,0]-np.array(right_imgpoints)[:,:,0,0])**2).mean())

    mae_camodolcal_3 = np.abs(np.array(left_imgpoints)[:,:,0,1]-np.array(right_imgpoints)[:,:,0,1]).mean()


    # print(f'pinhole n={n_pinhole} RMSE : {pinhole_rmse:.4f} MAE: {mae_pinhole:.4f}')
    # print(f'fisheye n={nfish} RMSE : {coner_rm_fisheye:.4f} MAE: {mae_fisheye:.4f}')
    # print(f'undis 1 n={n_undis_1} RMSE : {coner_rm_1:.4f} MAE: {mae_unidis_1:.4f}')
    # print(f'undis 3 n={n_undis_3} RMSE : {coner_rm_3:.4f} MAE: {mae_unidis_3:.4f}')
    # print(f'omidir  n={n_omidir} RMSE : {omidir_rmse:.4f} MAE: {mae_omidir:.4f}')
    print(f'camodolcal_3  n={n_camodolcal_3} RMSE : {camodolcal_3_rmse:.4f} MAE: {mae_camodolcal_3:.4f}')
    
    
    