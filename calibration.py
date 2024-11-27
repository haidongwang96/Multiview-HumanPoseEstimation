
import camera

def calibrate_both_intrinsics(sample_folder, cam_ids=[0,1]):
    """
    依次校准同一次拍摄的相机 0&1 的内参
    :param sample_folder: 输入两个相机同时拍摄的文件夹
    """

    for cam_id in cam_ids:
        calibrate_single_intrinscis(sample_folder, cam_id)
        calibrate_single_intrinscis(sample_folder, cam_id)

def calibrate_single_intrinscis(sample_folder, cam_id):
    """
    校准单个相机(cam_id)的内参
    """

    mtx, dist = camera.single_camera_calibrate_intrinsic_redo_with_rmse(sample_folder,cam_id)
    camera.save_camera_intrinsics(mtx, dist, f'camera_{cam_id}', option="json")



if __name__ == '__main__':
    sample_folder = "data/record_ubuntu/chessboard_0"
    calibrate_both_intrinsics(sample_folder,cam_ids=[2,4])
    # calibrate_single_intrinscis(sample_folder, 0)
    # calibrate_single_intrinscis(sample_folder,1)


