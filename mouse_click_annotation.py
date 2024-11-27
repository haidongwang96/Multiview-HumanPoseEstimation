import os

import cv2
import glob
import utility as su


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    img, click_index, click_coord = param
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(x,y)
        click_coord.append([x,y])
        cv2.circle(img, (x, y), 3, (255, 0, 0), thickness=-1)
        cv2.putText(img, str(click_index[0]), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 255, 0), thickness=1)
        cv2.imshow("image", img)
        click_index[0] += 1

def process(path, save_dir_path):
    img = cv2.imread(path)
    cv2.namedWindow("image")
    cv2.imshow("image", img)
    click_index = [0]
    click_coord = []
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, (img, click_index, click_coord))

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            fname = path.split("/")[-1]
            save_img_path = f"{save_dir_path}/{fname}"
            cv2.imwrite(save_img_path, img)
            print(f"Image saved to {save_img_path}")
            fname_txt = fname.replace(".jpg", ".txt")
            save_coord_path = f"{save_dir_path}/{fname_txt}"
            su.write_list_file(click_coord, save_coord_path)
            print(f"annotated coordinates saved to {save_img_path}")
            break


    cv2.destroyAllWindows()

if __name__ == '__main__':
    """
    脚本用于标定区域场景
    """

    cam_ids = [2,4]
    data_save_dir = "data/annotation/mouse_click"
    save_dir_path = su.create_asending_folder(data_save_dir,prefix="landmark")

    image_dir ="data/record_ubuntu/landmark_0"
    cam0_paths = glob.glob(f"{image_dir}/*_{cam_ids[0]}.jpg")
    cam1_paths = glob.glob(f"{image_dir}/*_{cam_ids[1]}.jpg")

    # for path0, path1 in zip(cam0_paths, cam1_paths):
    #
    #     process(path0, save_dir_path)
    #     process(path1, save_dir_path)

    process(cam0_paths[0], save_dir_path)
    process(cam1_paths[0], save_dir_path)


