
import utility as su
import camera

import numpy as np

cal = True

if cal:

    sample_folder = "record/sample_0"

    images_prefix_0 =f"{sample_folder}/*_0.jpg"
    cmtx0, dist0 = camera.calibrate_camera_for_intrinsic_parameters(images_prefix_0)
    camera.save_camera_intrinsics(cmtx0, dist0, 'camera0', option="json")

    images_prefix_1 =f"{sample_folder}/*_1.jpg"
    cmtx1, dist1 = camera.calibrate_camera_for_intrinsic_parameters(images_prefix_1)
    camera.save_camera_intrinsics(cmtx1, dist1, 'camera1', option="json")


    R, T = camera.stereo_calibrate(cmtx0, dist0, cmtx1, dist1, images_prefix_0, images_prefix_1)
    #print(R,T)
    # camera0 rotation and translation is identity matrix and zeros vector
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))
    #this will write R and T to disk
    camera.save_extrinsic_calibration_parameters(R0, T0, R, T, option="json")

else:
    # load pre-calculated parameters
    cmtx0,dist0 = camera.load_intrinsic_calibration_parameters("camera_parameters/camera0_intrinsics.json")
    cmtx1,dist1 = camera.load_intrinsic_calibration_parameters("camera_parameters/camera1_intrinsics.json")

    R0,T0 = camera.load_extrinsic_calibration_parameters("camera_parameters/camera0_extrinsics.json")
    R1,T1 = camera.load_extrinsic_calibration_parameters("camera_parameters/camera1_extrinsics.json")

    #check your calibration makes sense
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]

    camera.check_calibration('camera0', camera0_data, 'camera1', camera1_data, calibration_settings, _zshift=80.)


    import test_trivision
    #test_trivision.triangulate(cmtx0, cmtx1, R1, T1)

