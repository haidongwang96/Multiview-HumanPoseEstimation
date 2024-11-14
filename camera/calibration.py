
import camera

def calibrate_both_intrinsics(sample_folder):
    """
    依次校准同一次拍摄的相机 0&1 的内参
    :param sample_folder: 输入两个相机同时拍摄的文件夹
    """

    for cam_id in [0,1]:
        cmtx, dist = camera.single_camera_calibrate_intrinsic_parameters(sample_folder,cam_id)
        camera.save_camera_intrinsics(cmtx, dist, f'camera_{cam_id}', option="json")

def calibrate_single_intrinscis(sample_folder, cam_id):
    """
    校准单个相机(cam_id)的内参
    """

    cmtx, dist = camera.single_camera_calibrate_intrinsic_parameters(sample_folder, cam_id)
    camera.save_camera_intrinsics(cmtx, dist, f'camera_{cam_id}', option="json")



if __name__ == '__main__':
    sample_folder = "record/sample_1"

    # calibrate_both_intrinsics(sample_folder)

    calibrate_single_intrinscis(sample_folder,1)