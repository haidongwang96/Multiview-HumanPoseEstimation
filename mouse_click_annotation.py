import os

import cv2

import utility as su

# 1714402508613_0.jpg 403 248
# 1714402508613_1.jpg 257 316



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




data_save_dir = "data/annotation"
save_dir_path = su.create_asending_folder(data_save_dir,prefix="mouse_click")

image_dir ="data/record/aruco_1"
time_stamp = "1732521891855"

for idx in [0,1]:
    img_path = f"{image_dir}/{time_stamp}_{idx}.jpg"
    img = cv2.imread(img_path)
    cv2.namedWindow("image")
    cv2.imshow("image", img)
    click_index = [0]
    click_coord = []
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, (img,click_index,click_coord))

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            save_img_path = f"{save_dir_path}/{time_stamp}_{idx}.jpg"
            cv2.imwrite(save_img_path, img)
            print(f"Image saved to {save_img_path}")
            save_coord_path = f"{save_dir_path}/{time_stamp}_{idx}.txt"
            su.write_list_file(click_coord,save_coord_path)
            print(f"annotated coordinates saved to {save_img_path}")
            break
    cv2.destroyAllWindows()