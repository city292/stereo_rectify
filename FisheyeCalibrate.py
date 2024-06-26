import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
# import yaml

class FisheyeCalibrator():
    def __init__(self,img_dic,image_size):
        self._image_size = image_size[::-1]             # Image size                        : (height, width)
        self._board = None                  # Baord shape                       : (rows, cols)
        self._square_size = None            # Board square size in meter        : float
        self._ret = None                    # Stereo calibration return         : float
        self._left_K = np.zeros((3,3))      # Left camera matrix                : 3*3
        self._right_K = np.zeros((3,3))     # Right camera matrix               : 3*3
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
        self.left_images = [img_dic[id]['left'] for id in self.names]
        self.right_images = [img_dic[id]['right'] for id in self.names]
        self.object_points = [img_dic[id]['object_point'] for id in self.names]
        self.left_corners = [img_dic[id]['left_corner'] for id in self.names]
        self.right_corners = [img_dic[id]['right_corner'] for id in self.names]
        

    def stereo_calibrate(self,detail=True):
        # images -> calibrate params(left K, right K, left D, right D)

        # print(self._image_size)


        print("stereo calibrating ...")
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        self._rvec = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(self.names))]
        self._tvec = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(self.names))]
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 200, 0.01)
        flags = 0
        flags +=  cv2.fisheye.CALIB_FIX_SKEW 
        ret, self._right_K, self._right_D,_,_  = cv2.fisheye.calibrate(
            objectPoints=self.object_points,
            imagePoints=self.right_corners,
            image_size=self._image_size,
            K=self._right_K,
            D=self._right_D,
            rvecs=self._rvec,
            tvecs=self._tvec,
            # flags=flags,
            # criteria=criteria
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW 
            # + cv2.fisheye.CALIB_FIX_SKEW 
        )
        print(f'cv2.fisheye.calibrate: {ret}')
        
        print(self._right_K)
        print(self._right_D)
        # (objectPoints: Sequence[MatLike], imagePoints: Sequence[MatLike], image_size: Size, K: MatLike, D: MatLike, rvecs: Sequence[MatLike] | None = ..., tvecs: Sequence[MatLike] | None = ..., flags: int = ..., criteria: TermCriteria = ...) -> tuple[float, MatLike, MatLike, Sequence[MatLike], Sequence[MatLike]]
        ret, self._left_K, self._left_D,_,_  = cv2.fisheye.calibrate(
            objectPoints=self.object_points,
            imagePoints=self.left_corners,
            image_size=self._image_size,
            K=self._left_K,
            D=self._left_D,
            rvecs=self._rvec,
            tvecs=self._tvec,
            # flags=0,
            # criteria=criteria
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW 
            # + cv2.fisheye.CALIB_FIX_SKEW 
        )
        print(f'cv2.fisheye.calibrate: {ret}')
        
        print(self._left_K)
        print(self._left_D)

        flags=0
        flags += cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        # flags += cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
        flags += cv2.fisheye.CALIB_FIX_INTRINSIC
        flags +=  cv2.fisheye.CALIB_FIX_SKEW
        # flags += cv2.fisheye.CALIB_FIX_INTRINSIC
        (ret, self._left_K, self._left_D, self._right_K, self._right_D, self._R, self._T, self._rvec, self._tvec)  = cv2.fisheye.stereoCalibrate(
            objectPoints=self.object_points,
            imagePoints1=self.left_corners,
            imagePoints2=self.right_corners,
            K1=self._left_K,
            D1=self._left_D,
            K2=self._right_K,
            D2=self._right_D,
            R=self._R,
            T=self._T,
            imageSize=self._image_size,
            flags = flags,
            # criteria=criteria

        )
        print(f'cv2.fisheye.stereoCalibrate: {ret}')
        print(self._left_K)
        print(self._left_D)
        print(self._right_K)
        print(self._right_D)
        # print('8x8x8x'*10)

        print(self._R)
        print(self._T)
        # print(K1, self._left_K)
        # print(K2, self._right_K)
        # print(D1, self._left_D)
        # print(D2, self._right_D)
        # print('8888xxx'*10)

        # print(len(ret))
        # tuple[float, MatLike, MatLike, MatLike, MatLike, MatLike, MatLike, Sequence[MatLike], Sequence[MatLike]]
        print("stereo calibration was successful.")
        if detail:
            self.show_stereo_params()
        stereo_params_dict = {
            "left_K": self._left_K,
            "right_K": self._right_K,
            "left_D": self._left_D,
            "right_D": self._right_D,
            "rvec": self._rvec,
            # "rvec": self._R,
            "tvec": self._tvec
            # "tvec": self._T
        }
        return stereo_params_dict

    def stereo_rectify(self, detail=False):
        # _ -> rectify params(left R, right R, left P, right R, Q)
        print("stereo rectify...")
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        self._left_R, self._right_R, self._left_P, self._right_P, self._Q = cv2.fisheye.stereoRectify(
            K1=self._left_K,
            D1=self._left_D,
            K2=self._right_K,
            D2=self._right_D,
            imageSize=self._image_size,
            R=self._R,
            tvec=self._T,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
            newImageSize=(int(self._image_size[0]*self._mag), int(self._image_size[1]*self._mag)),
            fov_scale=1,
            balance=self.balance
        )
        print("stereo rectification was successful.")
        if detail:
            self.show_rectify_params()
        rectify_params_dict = {
            "left_R": self._left_R,
            "right_R": self._right_R,
            "left_P": self._left_P,
            "right_P": self._right_P,
            "Q": self._Q
        }
        newk = np.array([
            [self._image_size[0]/4,0,self._image_size[0]/2],
            [0,self._image_size[1]/4,self._image_size[1]/2],
            [0,0,1]
        ]
        )
        self._left_P= newk.copy()
        # self._left_P[1,1] = self._image_size[0]/4
        self._right_P= newk.copy()
        # self._right_P[1,1] = self._image_size[0]/4
        return rectify_params_dict

    def create_rectify_map(self):
        # _ -> rectify maps(left x, left y, right x, right y)
        self._left_rectify_map_x, self._left_rectify_map_y = cv2.fisheye.initUndistortRectifyMap(
            K=self._left_K,
            D=self._left_D,
            R=self._left_R,
            P=self._left_P,
            size=(int(self._image_size[0]*self._mag), int(self._image_size[1]*self._mag)),
            # size=self._image_size,
            m1type=cv2.CV_32FC1
        )
        self._right_rectify_map_x, self._right_rectify_map_y = cv2.fisheye.initUndistortRectifyMap(
            K=self._right_K,
            D=self._right_D,
            R=self._right_R,
            P=self._right_P,
            size=(int(self._image_size[0]*self._mag), int(self._image_size[1]*self._mag)),
            # size=self._image_size,
            m1type=cv2.CV_32FC1
        )
        print("rectify lookup table ceraeted.")
        rectify_map_dict = {
            "left_map_x": self._left_rectify_map_x,
            "left_map_y": self._left_rectify_map_y,
            "right_map_x": self._right_rectify_map_x,
            "right_map_y": self._right_rectify_map_y
        }
        self.show_rectify_params()
        return rectify_map_dict

    def create_equirectangular_map(self, axis="horizontal", mag_x=1.0, mag_y=1.0):
        # _ -> eqrec maps(left x, left y, right x, right y)
        assert axis in ["horizontal", "vertical"], "Select horizontal or vertical as axis."
        w,h = self._image_size
        w = int(w * mag_x)
        h = int(h * mag_y)
        self._left_eqrec_map_x = np.zeros((h,w), dtype=np.float32)
        self._left_eqrec_map_y = np.zeros((h,w), dtype=np.float32)
        self._right_eqrec_map_x = np.zeros((h,w), dtype=np.float32)
        self._right_eqrec_map_y = np.zeros((h,w), dtype=np.float32)
        # left map
        self.newk = np.array([
            [self._image_size[0]/4,0,self._image_size[0]/2],
            [0,self._image_size[1]/4,self._image_size[1]/2],
            [0,0,1]
        ]
        )
        fx = self.newk[0,0]
        fy = self.newk[1,1]
        cx = self.newk[0,2]
        cy = self.newk[1,2]
        # fx = 
        for y in range(h):
            for x in range(w):
                if axis=="vertical":
                    lamb = (1.0 - y/(h/2.0)) * (math.pi/2.0)
                    phi = (x/(w/2.0) - 1.0) * (math.pi/2.0)
                    vs_y = math.tan(lamb)
                    vs_x = math.tan(phi) / math.cos(lamb)
                    rec_x = cx + vs_x*fx
                    rec_y = cy - vs_y*fy
                    self._left_eqrec_map_x[y,x] = rec_x
                    self._left_eqrec_map_y[y,x] = rec_y
                elif axis=="horizontal":
                    lamb = (1.0 - x/(w/2.0)) * (math.pi/2.0)
                    phi = (1.0 + y/(h/2.0)) * (math.pi/2.0)
                    vs_x = math.tan(lamb)
                    vs_y = math.tan(phi) / math.cos(lamb)
                    rec_x = cx - vs_x*fx
                    rec_y = cy + vs_y*fy
                    self._left_eqrec_map_x[y,x] = rec_x
                    self._left_eqrec_map_y[y,x] = rec_y
        # right map
        fx = self.newk[0,0]
        fy = self.newk[1,1]
        cx = self.newk[0,2]
        cy = self.newk[1,2]
        for y in range(h):
            for x in range(w):
                if axis=="vertical":
                    lamb = (1.0 - y/(h/2.0)) * (math.pi/2.0)
                    phi = (x/(w/2.0) - 1.0) * (math.pi/2.0)
                    vs_y = math.tan(lamb)
                    vs_x = math.tan(phi) / math.cos(lamb)
                    rec_x = cx + vs_x*fx
                    rec_y = cy - vs_y*fy
                    self._right_eqrec_map_x[y,x] = rec_x
                    self._right_eqrec_map_y[y,x] = rec_y
                elif axis=="horizontal":
                    lamb = (1.0 - x/(w/2.0)) * (math.pi/2.0)
                    phi = (1.0 + y/(h/2.0)) * (math.pi/2.0)
                    vs_x = math.tan(lamb)
                    vs_y = math.tan(phi) / math.cos(lamb)
                    rec_x = cx - vs_x*fx
                    rec_y = cy + vs_y*fy
                    self._right_eqrec_map_x[y,x] = rec_x
                    self._right_eqrec_map_y[y,x] = rec_y
        print("equirectangular lookup table ceraeted.")
        eqrec_map_dict = {
            "left_map_x": self._left_eqrec_map_x,
            "left_map_y": self._left_eqrec_map_y,
            "right_map_x": self._right_eqrec_map_x,
            "right_map_y": self._right_eqrec_map_y
        }
        return eqrec_map_dict

    def get_rectified_image(self, left_image,right_image, save=None, show=True):
        # image -> rectified_image
        msg = "Plz create map first."
        assert self._left_rectify_map_x is not None, msg
        assert self._left_rectify_map_y is not None, msg
        assert self._right_rectify_map_x is not None, msg
        assert self._right_rectify_map_y is not None, msg
        # left_image, right_image = self._split_image(image)

        left_rectified_image = cv2.remap(left_image, self._left_rectify_map_x, self._left_rectify_map_y, cv2.INTER_LINEAR)
        right_rectified_image = cv2.remap(right_image, self._right_rectify_map_x, self._right_rectify_map_y, cv2.INTER_LINEAR)
        rectified_image = np.concatenate([left_rectified_image, right_rectified_image], axis=1)
        if save is not None:

            for y in range(0,rectified_image.shape[0],50):
                cv2.line(rectified_image,(0,y),(rectified_image.shape[1],y),(0,0,255))
            cv2.imwrite(save, rectified_image)

        return rectified_image, left_rectified_image, right_rectified_image

    def get_equirectangular_image(self, left_image,right_image, save=None, show=True):
        # image -> rectified_image
        msg = "Plz create map first."
        assert self._left_eqrec_map_x is not None, msg
        assert self._left_eqrec_map_y is not None, msg
        assert self._right_eqrec_map_x is not None, msg
        assert self._right_eqrec_map_y is not None, msg
        # left_image, right_image = self._split_image(image)
        left_eqrec_image = cv2.remap(left_image, self._left_eqrec_map_x, self._left_eqrec_map_y, cv2.INTER_LINEAR)
        right_eqrec_image = cv2.remap(right_image, self._right_eqrec_map_x, self._right_eqrec_map_y, cv2.INTER_LINEAR)
        eqrec_image = np.concatenate([left_eqrec_image, right_eqrec_image], axis=1)
        if save is not None:
            # print(eqrec_image.shape)
            for j in range(20):
                cv2.line(eqrec_image, pt1=(0, 54 * j), pt2=(1352 * 2 - 1, 54 * j), color=(0, 0, 255), thickness=2)

            cv2.imwrite(save, eqrec_image)
        return eqrec_image,left_eqrec_image,right_eqrec_image

    def show_stereo_params(self):
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print("stereo params")
        print("###################################")
        print(f"Return: {self._ret}")
        print(f"Left K:\n{self._left_K}")
        print(f"Left D:\n{self._left_D}")
        print(f"Right K:\n{self._right_K}")
        print(f"Right D:\n{self._right_D}")
        # print(f"R vector:\n{self._rvec}")
        # print(f"T vector:\n{self._tvec}")
        print("###################################")

    def show_rectify_params(self):
        print("rectify params")
        print("###################################")
        print("Left R:\n",self._left_R)
        print("Right R:\n",self._right_R)
        print("Left P:\n",self._left_P)
        print("Right P:\n",self._right_P)
        print("Q:\n",self._Q)
        print("###################################")

    def _split_image(self, image):
        # image -> left_half, right_half
        left_image = image[:,:image.shape[1]//2]
        right_image = image[:, image.shape[1]//2:]
        return left_image, right_image

    def _create_object_point(self):
        # _ -> object points grid refered board
        object_point = np.zeros((1, self._board[0]*self._board[1], 3), np.float32)
        object_point[0,:,:2] = np.mgrid[0:self._board[0], 0:self._board[1]].T.reshape(-1, 2)
        object_point = object_point*self._square_size 
        return object_point

    def _detect_corners(self, image):
        # image -> left half corners point, right corners point
        left_image, right_image = self._split_image(image)
        flag = cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        left_ret, left_corner = cv2.findChessboardCorners(left_image, self._board, flag)
        # print(left_ret, left_corner)
        if not left_ret:
            print('left_ret fali')
            return [], []
        if len(left_corner)>0:
            for points in left_corner:
                cv2.cornerSubPix(left_image, points, winSize=(3, 3), zeroZone=(-1,-1), criteria=criteria)
        right_ret, right_corner = cv2.findChessboardCorners(right_image, self._board, flag)
        # print(right_ret, right_corner)
        if not right_ret:
            print('right_ret fali')
            return [], []
        if len(right_corner)>0:
            for points in right_corner:
                cv2.cornerSubPix(right_image, points, winSize=(3, 3), zeroZone=(-1,-1), criteria=criteria)
        return left_corner, right_corner

    def _show_images(self, images, titles, figsize, subplot):
        # images -> show images
        plt.figure(figsize=figsize)
        for i, (image, title) in enumerate(zip(images, titles)):
            plt.subplot(subplot[0], subplot[1], i+1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis("off")
        # plt.show()
        # plt.imsave('/data/temp/tmp.png')
        print('save ---'*10)
        plt.savefig('/data/temp/tmp.png')
        