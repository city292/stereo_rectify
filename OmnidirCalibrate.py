import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
# import yaml
from pathlib import Path
class OmnidirCalibrator():
    def __init__(self,img_dic,image_size):
        self._image_size = image_size[::-1]             # Image size                        : (height, width)
        self._board = None                  # Baord shape                       : (rows, cols)
        self._square_size = None            # Board square size in meter        : float
        self._ret = None                    # Stereo calibration return         : float
        self._left_K = np.zeros((3,3))      # Left camera matrix                : 3*3
        self._right_K = np.zeros((3,3))     # Right camera matrix               : 3*3
        self._left_xi = None
        self._right_xi = None
        self._left_D = np.zeros((4,1))      # Left camera distortion vector     : 4*1
        self._right_D = np.zeros((4,1))     # Right camera distortion vector    : 4*1
        self._R = np.zeros((3,3))
        self._T = np.zeros((1,3))
        self._rvec = None                   # cameras rotation vector           : 1*1*3
        self._tvec = None                   # cameras translation vector        : 1*1*3
        self._left_R = np.zeros((3,3))      # Left camera rotation matrix       : 3*3
        self._left_P = np.zeros((3,4))      # Left camera projection matrix     : 3*4
        self._right_R = np.zeros((3,3))     # Right camera rotation matrix      : 3*3
        self._right_P = np.zeros((3,4))     # Right camera projection matrix    : 3*4
        self._Q = np.zeros((4,4))           # Disparity-to-depth mapping matrix : 4*4
        self._left_rectify_map_x = None     # Left rectify lookup table x       : height*width
        self._left_rectify_map_y = None     # Left rectify lookup table y       : height*width   
        self._right_rectify_map_x = None    # Right rectify lookup table x      : height*width
        self._right_rectify_map_y = None    # Right rectify lookup table y      : height*width
        self._left_eqrec_map_x = None       # Left eqrec lookup table x         : height*width
        self._left_eqrec_map_y = None       # Left eqrec lookup table y         : height*width   
        self._right_eqrec_map_x = None      # Right eqrec lookup table x        : height*width
        self._right_eqrec_map_y = None      # Right eqrec lookup table y        : height*width
        self._mag = 1.0                     # Fov related prams in rectify      : float
        self.balance = 1.0
        self.img_dic = img_dic
        self._square_size = 0.52
        self.names = list(img_dic.keys())
        self.names = sorted(self.names)
        self.rvecsL =None
        self.tvecsL = None
        self.left_images = [img_dic[id]['left'] for id in self.names]
        self.right_images = [img_dic[id]['right'] for id in self.names]
        self.object_points = [img_dic[id]['object_point'] for id in self.names]
        self.left_corners = [img_dic[id]['left_corner'] for id in self.names]
        self.right_corners = [img_dic[id]['right_corner'] for id in self.names]
        

    def stereo_calibrate(self,detail=True):
        # images -> calibrate params(left K, right K, left D, right D)

        print("stereo calibrating ...")
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        self._rvec = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(self.names))]
        self._tvec = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(self.names))]

        # (objectPoints: Sequence[MatLike], imagePoints: Sequence[MatLike], image_size: Size, K: MatLike, D: MatLike, rvecs: Sequence[MatLike] | None = ..., tvecs: Sequence[MatLike] | None = ..., flags: int = ..., criteria: TermCriteria = ...) -> tuple[float, MatLike, MatLike, Sequence[MatLike], Sequence[MatLike]]
        flags = 0
        # flags += cv2.omnidir.CALIB_FIX_SKEW
        # flags += cv2.omnidir.
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 200, 0.001)

        ret, self._right_K, self._right_xi, self._right_D,self._rvec,self._tvec,idx  = cv2.omnidir.calibrate(
            objectPoints=self.object_points,
            imagePoints=self.right_corners,
            size=self._image_size,
            K=self._right_K,
            D=None,
            xi=None,
            flags=flags,
            criteria=criteria
            )
        print(f'cv2.omnidir.calibrate: right:{ret}')
        print(idx)
        ret, self._left_K, self._left_xi, self._left_D,self._rvec,self._tvec,idx  = cv2.omnidir.calibrate(
            objectPoints=self.object_points,
            imagePoints=self.left_corners,
            size=self._image_size,
            K=self._left_K,
            D=None,
            xi=None,
            flags=flags,
            criteria=criteria
            )

        print(f'cv2.omnidir.calibrate: left:{ret}')
        print(idx)
        print(self._left_K)
        print(self._left_D)
        print(self._right_K)
        print(self._right_D)
        print(self._left_xi, self._right_xi)
        print(len(self.object_points),len(self.left_corners),len(self.right_corners))
        flags = 0 
        flags += cv2.omnidir.CALIB_USE_GUESS
        ret, ops, ips1, ips2, self._left_K, self._left_xi, self._left_D, self._right_K, self._right_xi, self._right_D,self._R, self._T,self.rvecsL,self.tvecsL,idx  = cv2.omnidir.stereoCalibrate(
            objectPoints=self.object_points,
            imagePoints1=self.left_corners,
            imagePoints2=self.right_corners,
            imageSize1=self._image_size,
            imageSize2=self._image_size,
            K1=self._left_K,
            xi1=self._left_xi,
            D1=self._left_D,
            K2=self._right_K,
            xi2=self._right_xi,
            D2=self._right_D,
            flags=flags,
            criteria=criteria
            )
        print(f'cv2.omnidir.stereoCalibrate: {ret}')
        print(idx)
        print(self._left_K)
        print(self._left_D)
        print(self._right_K)
        print(self._right_D)
        print(self._left_xi, self._right_xi)
        print(self._R)
        print(self._T)

        # print('8888xxx'*10)

        # print(len(ret))
        # tuple[float, MatLike, MatLike, MatLike, MatLike, MatLike, MatLike, Sequence[MatLike], Sequence[MatLike]]
        print("stereo calibration was successful.")

        stereo_params_dict = {
            "left_K": self._left_K,
            "right_K": self._right_K,
            "left_D": self._left_D,
            "right_D": self._right_D,
            # "rvec": self._rvec,
            "rvec": self._R,
            # "tvec": self._tvec
            "tvec": self._T
        }
        return stereo_params_dict
    def rectify(self, image_left, image_right, save=None):
        Knew = self._left_K.copy()
        Knew[0,0] /=2
        Knew[1,1] /=2
        Knew[:2,2] =0
        out_left = cv2.omnidir.undistortImage(
            image_left,
            self._left_K,
            self._left_D,
            self._left_xi,
            # flags=cv2.omnidir.RECTIFY_PERSPECTIVE,
            flags=cv2.omnidir.RECTIFY_LONGLATI,
            Knew=Knew)
        Knew = self._right_K.copy()
        Knew[0,0] /=2
        Knew[1,1] /=2
        Knew[:2,2] =0
        out_right = cv2.omnidir.undistortImage(
            image_right,
            self._right_K,
            self._right_D,
            self._right_xi,
            flags=cv2.omnidir.RECTIFY_LONGLATI,
            Knew=Knew)
        rectified_image = np.concatenate([out_left, out_right], axis=1)
        if save is not None:

            for y in range(0,rectified_image.shape[0],50):
                cv2.line(rectified_image,(0,y),(rectified_image.shape[1],y),(0,0,255))
            cv2.imwrite(save, rectified_image)
            s = Path(save)
            n = s.name
            s = s.parents[1]/'left'/n
            s.parent.mkdir(parents=True,exist_ok=True)
            
            # for y in range(0,out_left.shape[0],50):
            #     cv2.line(out_left,(0,y),(out_left.shape[1],y),(0,0,255))
            cv2.imwrite(s.as_posix(), out_left)
            s = Path(save)
            n = s.name
            s = s.parents[1]/'right'/n
            # for y in range(0,out_right.shape[0],50):
            #     cv2.line(out_right,(0,y),(out_right.shape[1],y),(0,0,255))
            s.parent.mkdir(parents=True,exist_ok=True)
            
            cv2.imwrite(s.as_posix(), out_right)
        return out_left, out_right
        
        
    def stereo_rectify(self,  image_left, image_right, save=None):
        # _ -> rectify params(left R, right R, left P, right R, Q)
        disp, image_left_rec, image_right_rec, pointcloud =cv2.omnidir.stereoReconstruct(
            image1=image_left,
            image2=image_right,
            K1=self._right_K,
            D1=self._left_D,
            xi1=self._left_xi,
            K2=self._right_K,
            D2=self._right_D,
            xi2=self._right_xi,
            R=self._R,
            T=self._T,
            # newSize = 
            flag=cv2.omnidir.RECTIFY_PERSPECTIVE,
            numDisparities=16*5,
            SADWindowSize=5,
            Knew=np.array([
                [self._image_size[0]/4,0,self._image_size[0]/2],
                [0,self._image_size[1]/4,self._image_size[1]/2],
                [0,0,1],
                
            ])
            
            
        )
        rectified_image = np.concatenate([image_left_rec, image_right_rec], axis=1)
        if save is not None:

            for y in range(0,rectified_image.shape[0],50):
                cv2.line(rectified_image,(0,y),(rectified_image.shape[1],y),(0,0,255))
            cv2.imwrite(save, rectified_image)

            s = Path(save)
            n = s.name
            s = s.parents[1]/'left'/n
            # for y in range(0,image_left_rec.shape[0],50):
            #     cv2.line(image_left_rec,(0,y),(image_left_rec.shape[1],y),(0,0,255))
            s.parent.mkdir(parents=True,exist_ok=True)
            
            cv2.imwrite(s.as_posix(), image_left_rec)
        
            s = Path(save)
            n = s.name
            s = s.parents[1]/'right'/n
            s.parent.mkdir(parents=True,exist_ok=True)
            
            # for y in range(0,image_right_rec.shape[0],50):
            #     cv2.line(image_right_rec,(0,y),(image_right_rec.shape[1],y),(0,0,255))
            cv2.imwrite(s.as_posix(), image_right_rec)
        return rectified_image, image_left_rec, image_right_rec
    def rectifi_points(self, point):
        # cv2.omnidir.
        pass
        
        
        