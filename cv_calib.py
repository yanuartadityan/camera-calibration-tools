"""
cv_calib.py
"""

import os
import glob
import time
import yaml
import numpy as np
import cv2 as cv

# source
calib_vid = r"C:\Workspace\Git\camera-calibration-tools\calib_streams\p20_2cm.mp4"
calib_img = r"C:\Workspace\Git\camera-calibration-tools\calib_images"


# criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


class CvCalibTools():
    """a class with plenty of tools for camera calibrations"""
    @staticmethod
    def calib_rec_images(frame, pathname=os.path.dirname(os.path.realpath(__file__))):
        i = 0
        while os.path.exists(os.path.join(pathname, 'calib_img_%s.png' % i)):
            i += 1

        cv.imwrite(os.path.join(pathname, 'calib_img_%s.png' % i), frame)

    @staticmethod
    def calib_handpick_images(vidpath=None, imout=os.path.join(os.path.dirname(__file__), 'output')):

        # the input path pointing to the video input must exists
        if not os.path.exists(vidpath):
            raise ValueError

        # the output path where all chosen images shall be stored. must exists
        if not os.path.exists(imout):
            os.makedirs(imout)

        # load the video
        cap = cv.VideoCapture(vidpath)

        # create a window
        cv.namedWindow('Display')

        # iterate frames
        isPlaying = True
        while True:

            if isPlaying:
                ret, frame = cap.read()

                # display
                cv.imshow('Display', frame)

                # to mono
                grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # register keypress
            kpress = cv.waitKey(1)

            if kpress & 0xFF == ord('q'):
                break

            # pause/unpause
            if kpress == ord('s'):
                if isPlaying:
                    isPlaying = False
                else:
                    isPlaying = True

            # record
            if kpress == ord('r'):
                i = 0
                while os.path.exists(os.path.join(imout, 'calib_img_%s.png' % i)):
                    i += 1

                cv.imwrite(os.path.join(imout, 'calib_img_%s.png' % i), grayed)

    @staticmethod
    def calib_detect_chessboard(pathname=None, chessboard_size=(6, 8), square_size=20):
        images = glob.glob(os.path.join(pathname, '*.png'))

        # assign row and col as the inner corner dimension of the checkerboard, e.g., 6x8, 7x9, etc
        row = chessboard_size[0]
        col = chessboard_size[1]

        # allocate (row*col, 2) matrix which stores the coordinate of the checkerboard 
        # in the real world size
        objp = np.zeros((row * col, 3), np.float32)
        objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2) * square_size

        # print
        print('detecting %d number of calibration images' % len(images))

        # containers
        objpoints = []
        imgpoints = []

        # create window
        cv.namedWindow('Chessboard: Detection')

        samples = 0

        for fname in images:
            # read calib images and make sure they are coming in just one channel of color
            img = cv.imread(fname)

            original = img.copy()
            grayed = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(grayed, (col, row), None)

            if ret:
                # update object points
                objpoints.append(objp)

                # save corners
                corners2 = cv.cornerSubPix(grayed, corners, (11, 11), (-1, -1), criteria)

                # update image points
                imgpoints.append(corners2)

                # draw
                grayed = cv.drawChessboardCorners(original, (col, row), corners2, ret)

                # stack
                h_stack = np.hstack((img, grayed))
                h_stack = cv.resize(h_stack, None, fx=0.4, fy=0.4)

                cv.putText(h_stack, 'Samples captured: %d' % samples, (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv.imshow('Chessboard: Detection', h_stack)
                cv.waitKey(0)

                samples += 1

        # summary
        print('processed %d numbers of calibration images' % len(images))
        print('found %d-%d image-to-object correlations points' % (len(imgpoints), len(objpoints)))

        # destroy window
        cv.destroyAllWindows()

        # return
        return imgpoints, objpoints

    @staticmethod
    def calibrate_camera(pathname, image_points, object_points):
        images = glob.glob(os.path.join(pathname, '*.png'))

        # compute new optimal camera matrix
        init_img = cv.imread(images[0])

        # get shape (x, y) or (w, h)
        shape = init_img.shape[::-1][1:]

        # find the camera matrix
        lret, lmtx, ldist, lrvecs, ltvecs = cv.calibrateCamera(object_points, image_points, shape, None, None)

        # get optimal new camera matrix
        ncam_mtx, roi = cv.getOptimalNewCameraMatrix(lmtx, ldist, shape, 0, shape)

        # fname
        for fname in images:
            img = cv.imread(fname)

            # undistort
            undistorted = cv.undistort(img, lmtx, ldist, None, ncam_mtx)

            # show
            concatimg_horizontal = np.concatenate((img, undistorted), axis=1)
            cv.imshow('Undistorted Images', cv.resize(concatimg_horizontal, None, fx=0.4, fy=0.4))
            cv.waitKey(0)

        # save to a file
        dump_calib = {'camera_matrix': np.asarray(lmtx).tolist(),
                      'optimised_camera_matrix': np.asarray(ncam_mtx).tolist(),
                      'dist_coeff': np.asarray(ldist).tolist(),
                      'reprojection_err': lret}

        with open(os.path.join(pathname, 'calibration.yaml'), 'w') as f:
            yaml.dump(dump_calib, f)

        return True

    @staticmethod
    def read_calibration_data(pathname):
        with open(pathname) as f:
            loaded_dict = yaml.load(f)

        mtxloaded = loaded_dict.get('camera_matrix')
        distloaded = loaded_dict.get('dist_coeff')
        optmtxloaded = loaded_dict.get('optimised_camera_matrix')

        return mtxloaded, distloaded, optmtxloaded

    @staticmethod
    def read_png(pathname):
        images = glob.glob(os.path.join(pathname, '*.png'))

        for fname in images:
            img = cv.imread(fname)

            cv.imshow('%s' % fname, img)
            cv.waitKey(1)


class Test():
    """test
    """

    @staticmethod
    def test_calib(src=calib_img):
        img, obj = CvCalibTools.calib_detect_chessboard(src, (7, 9))
        ret = CvCalibTools.calibrate_camera(src, img, obj)

    @staticmethod
    def test_select_images(src=calib_vid):
        CvCalibTools.calib_handpick_images(vidpath=src, imout=r'C:\Workspace\Git\camera-calibration-tools\calib_images\huawei')


if __name__ == '__main__':

    # Test.test_select_images()
    Test.test_calib(src=r"C:\Workspace\Git\camera-calibration-tools\calib_images\huawei")
