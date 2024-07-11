import json
import os
import cv2
import pdb
import random
import numpy as np

'''
One row per object
Each row is class x_center y_center width height format.
Box coordinates must be in normalized xywh format (from 0 to 1). If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
Class numbers are zero-indexed (start from 0).
'''


def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def convert_bbox_to_yolo_format(bbox_obj, yolo_obj_id, img_width, img_height):
    x, y, w_bbox, h_bbox = bbox_obj

    x_center = x + w_bbox / 2
    y_center = y + h_bbox / 2

    x_center /= img_width
    y_center /= img_height
    w_bbox /= img_width
    h_bbox /= img_height

    yolo_format = f"{yolo_obj_id} {x_center} {y_center} {w_bbox} {h_bbox}"

    return yolo_format


def save_yolo_format_to_file(bbox_obj, yolo_obj_id, img_width, img_height, output_file):
    with open(output_file, 'w') as file:
        yolo_format = convert_bbox_to_yolo_format(bbox_obj, yolo_obj_id, img_width, img_height)
        file.write(yolo_format + '\n')

def add_random_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = random.uniform(0.001, 0.005)
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_image = image + gauss * 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def apply_random_gaussian_blur(image):
    ksize = random.choice([3, 5])
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred_image

def apply_random_motion_blur(image):
    size = random.randint(3, 7)
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    motion_blurred_image = cv2.filter2D(image, -1, kernel_motion_blur)
    return motion_blurred_image

def apply_random_effects(image):
    
    #randomly select if apply augmentation or not
    if random.uniform(0, 1) > 0.5:
            image = apply_random_gaussian_blur(image)
    else:
        image = apply_random_motion_blur(image)
    
    if random.uniform(0, 1) > 0.5:
        image = add_random_noise(image)
    
    #applay a random number of effects. this means that some images won't have any effect
    '''for _ in range(0, int(random.uniform(0, 4))):
        effect = random.choice(effects)
        image = effect(image)'''
    return image



# ----TO EDIT WITH THE CORRECT FOLDER ---------
BOP_dataset_folder = "datasets\\new"
YOLO_dataset_folder = "..\\YOLOdatasets\\laryngoscope"
YOLO_folder_postfix = "train"
YOLO_file_name_prefix = "sticker_augmented" # "no_stickers"
yolo_obj_id = 0  # id of the object of the yolo dataset
num_images = 1000
apply_augmentation = True
# __________________________

# Percorsi ai file JSON
scene_gt_filepath = f"{BOP_dataset_folder}\\000000\\scene_gt.json"
scene_gt_info_filepath = f"{BOP_dataset_folder}\\000000\\scene_gt_info.json"
base_image_path_BOP = f"{BOP_dataset_folder}\\000000\\rgb"
base_image_path_YOLO = f"{YOLO_dataset_folder}\\images\\{YOLO_folder_postfix}"
base_label_path_YOLO = f"{YOLO_dataset_folder}\\labels\\{YOLO_folder_postfix}"

os.makedirs(base_image_path_YOLO, exist_ok=True)
os.makedirs(base_label_path_YOLO, exist_ok=True)

scene_gt_data = load_json(scene_gt_filepath)
scene_gt_info_data = load_json(scene_gt_info_filepath)

i = 0
for key, info_data in scene_gt_info_data.items():
    if i >= num_images:
        break
    print(key)
    scene_id = '{:06d}'.format(int(key))
    image_file_path_BOP = base_image_path_BOP + f"\\{scene_id}.png"
    image_file_path_YOLO = base_image_path_YOLO + f"\\{YOLO_file_name_prefix}_{scene_id}.png"
    label_path_YOLO = base_label_path_YOLO + f"\\{YOLO_file_name_prefix}_{scene_id}.txt"
    image = cv2.imread(image_file_path_BOP)
    if apply_augmentation:
        image = apply_random_effects(image);

    info = info_data[0]
    h, w, c = image.shape
    # save label
    save_yolo_format_to_file(info["bbox_obj"], yolo_obj_id, w, h, label_path_YOLO)
    # save image
    cv2.imwrite(image_file_path_YOLO, image)

    i += 1
