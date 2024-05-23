import os
import sys
import pdb

sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from config_parser import parse_cfg
import argparse
import shutil
from tqdm import tqdm
from bop_dataset_pytorch import get_roi

from tools_for_BOP import bop_io
from bop_dataset_pytorch import bop_dataset_single_obj_pytorch

import torch
import numpy as np
import cv2

from binary_code_helper.CNN_output_to_pose import load_dict_class_id_3D_points, CNN_outputs_to_object_pose

sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout

from model.BinaryCodeNet import BinaryCodeNet_Deeplab

from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP

from get_detection_results import ycbv_select_keyframe
from common_ops import from_output_to_class_mask, from_output_to_class_binary_code, compute_original_mask
from tools_for_BOP.common_dataset_info import get_obj_info

from binary_code_helper.generate_new_dict import generate_new_corres_dict

from tools_for_BOP import write_to_cvs

import torchvision.transforms as transforms

from PIL import Image

images_to_show = dict()

# ----------plot defs----------------------------------------
vertices = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [1, 2, 0],
                     [0, 2, 0],
                     [0, 0, 1],
                     [1, 0, 1],
                     [1, 2, 1],
                     [0, 2, 1]])

edges = [[0, 1], [1, 2], [2, 3], [3, 0],
         [4, 5], [5, 6], [6, 7], [7, 4],
         [0, 4], [1, 5], [2, 6], [3, 7]]


def show_image(name, image):
    images_to_show[name] = image


def update_plot(R, T, image):
    plt.clf()  # Pulisce la figura precedente
    num_images = len(images_to_show)
    num_cols = min(num_images, 3)  # Numero di colonne nel subplot
    num_rows = int(np.ceil(num_images / num_cols)) + 1  # Calcola il numero di righe nel subplot

    for i, (name, image) in enumerate(images_to_show.items()):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(name)

    plot = plt.subplot(num_rows, num_cols, len(images_to_show) + 1, projection='3d')
    show_rotation_translation_plot(R, T, plot, image)

    plt.tight_layout()


def _do_transformation(T, R):
    return np.dot(vertices.copy() - [0.5, 0.5, 0.5], R.T) + [0.5, 0.5, 0.5] + T


def show_rotation_translation_plot(R, T, plot, image):
    transformed_vertices = _do_transformation(T, R)
    plt.cla()
    # Disegna i vertici trasformati
    plot.scatter3D(transformed_vertices[:, 0], transformed_vertices[:, 1], transformed_vertices[:, 2])

    # Disegna i bordi trasformati
    for edge in edges:
        plot.plot3D(*zip(*transformed_vertices[edge]), color='b')
    # ---------------
    '''# Disegna i vertici originali
    plot.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    # Disegna i bordi originali
    for edge in edges:
        plot.plot3D(*zip(*vertices[edge]), color='g', alpha=0.2)'''

    # Imposta i limiti dell'asse per corrispondere alle dimensioni dell'immagine
    #plot.set_xlim([0, image.shape[1]])
    #plot.set_ylim([0, image.shape[0]])
    #plot.set_zlim([0, max(image.shape)])
    #plt.imshow(image)

    plot.set_aspect('equal', 'box')

    plt.draw()
    plt.pause(0.0001)


# --------------------------------------


def read_depth(path):
    depth = np.asarray(Image.open(path)).copy()
    depth = depth.astype(np.float64)
    return depth


def convert_image_to_tensor(rgbImage):
    composed_transforms_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    x_pil = Image.fromarray(np.uint8(rgbImage)).convert('RGB')
    return composed_transforms_img(x_pil)


def main(configs):
    def process(plt_anim_frame):

        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            return

        # --------------------predict------------------------------------------------------------------
        test_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gray_image = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        result_image = cv2.merge((gray_image, gray_image, gray_image))
        test_img = result_image

        # find bbox
        # per test questi bbox sono falsi
        b_box = [200, 40, 140, 140]
        # test_img = cv2.cvtColor(cv2.imread("1.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        roi_x = get_roi(test_img, b_box, BoundingBox_CropSize_image, interpolation=cv2.INTER_LINEAR,
                        resize_method=resize_method)

        # show_image("test", roi_x.copy())
        x, y, w, h = b_box
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        show_image("test", test_img)

        data = convert_image_to_tensor(roi_x).unsqueeze(0)

        if torch.cuda.is_available():
            data = data.cuda()

        pred_mask_prob, pred_code_prob = net(data)

        pred_masks = from_output_to_class_mask(pred_mask_prob)
        pred_code_images = from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type,
                                                            divided_num_each_interation=divide_number_each_itration,
                                                            binary_code_length=binary_code_length)

        # from binary code to pose
        pred_code_images = pred_code_images.transpose(0, 2, 3, 1)

        pred_masks = pred_masks.transpose(0, 2, 3, 1)
        pred_masks = pred_masks.squeeze(axis=-1).astype('uint8')

        show_image("pred mask", pred_masks[0] * 255)

        R = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        T = np.zeros([1, 3])

        R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[0],
                                                                   pred_code_images[0],
                                                                   b_box, BoundingBox_CropSize_GT,
                                                                   divide_number_each_itration,
                                                                   dict_class_id_3D_points,
                                                                   intrinsic_matrix=cam_K)

        if success:
            R = R_predict
            T = t_predict
        # print(R_predict, t_predict)
        # image_with_indicator = draw_indicator(frame, R_predict, t_predict.flatten())
        # ------------------------------------------------------------------------------------------------
        update_plot(R, T.flatten(), test_img)  # T.flatten()

    # ---------- -------- ---------- ---------- ----------- ----------- ---------
    #### training dataset
    bop_challange = configs['bop_challange']
    bop_path = configs['bop_path']
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    training_data_folder = configs['training_data_folder']
    training_data_folder_2 = configs['training_data_folder_2']
    test_folder = configs['test_folder']  # usually is 'test'
    second_dataset_ratio = configs['second_dataset_ratio']  # the percentage of second dataset in the batch
    num_workers = configs['num_workers']
    train_obj_visible_theshold = configs[
        'train_obj_visible_theshold']  # for test is always 0.1, for training we can set different values, usually 0.2
    #### network settings
    BoundingBox_CropSize_image = configs['BoundingBox_CropSize_image']  # input image size
    BoundingBox_CropSize_GT = configs['BoundingBox_CropSize_GT']  # network output size
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']  # now only support "L1" or "BCE"
    output_kernel_size = configs['output_kernel_size']  # last layer kernel size
    resnet_layer = configs['resnet_layer']  # usually resnet 34
    concat = configs['concat_encoder_decoder']
    efficientnet_key = configs['efficientnet_key']
    resize_method = configs['resize_method']

    # pixel code settings
    divide_number_each_itration = configs['divide_number_each_itration']
    number_of_itration = configs['number_of_itration']

    # cam_K = cam_params[obj_id][0]['cam_K'].reshape((3, 3))
    cam_K = [[1.50557939e+03, 0.00000000e+00, 2.91158435e+02],
             [0.00000000e+00, 1.47186344e+03, 1.02544916e+02],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

    # get dataset informations
    dataset_dir, source_dir, model_plys, model_info, model_ids, rgb_files, depth_files, mask_files, mask_visib_files, gts, gt_infos, cam_param_global, cam_params = bop_io.get_dataset(
        bop_path, dataset_name, train=True, data_folder=training_data_folder, data_per_obj=True, incl_param=True,
        train_obj_visible_theshold=train_obj_visible_theshold)
    obj_name_obj_id, symmetry_obj = get_obj_info(dataset_name)
    obj_id = int(obj_name_obj_id[obj_name] - 1)  # now the obj_id started from 0
    if obj_name in symmetry_obj:
        Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADD_Error_BOP
        main_metric_name = 'ADI'
        supp_metric_name = 'ADD'
    else:
        Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADI_Error_BOP
        main_metric_name = 'ADD'
        supp_metric_name = 'ADI'

    path_dict = os.path.join(dataset_dir, "models_GT_color", "Class_CorresPoint{:06d}.txt".format(obj_id + 1))
    total_numer_class, _, _, dict_class_id_3D_points = load_dict_class_id_3D_points(path_dict)
    divide_number_each_itration = int(divide_number_each_itration)
    total_numer_class = int(total_numer_class)
    number_of_itration = int(number_of_itration)
    if divide_number_each_itration ** number_of_itration != total_numer_class:
        raise AssertionError("the combination is not valid")
    GT_code_infos = [divide_number_each_itration, number_of_itration, total_numer_class]

    if divide_number_each_itration != 2 and (BinaryCode_Loss_Type == 'BCE' or BinaryCode_Loss_Type == 'L1'):
        raise AssertionError("for non-binary case, use CE as loss function")
    if divide_number_each_itration == 2 and BinaryCode_Loss_Type == 'CE':
        raise AssertionError("not support for now")

    torch.manual_seed(0)  # the both are only good for ablation study
    np.random.seed(0)  # if can be removed in the final experiments

    calc_add_and_adi = True

    binary_code_length = number_of_itration
    print("predicted binary_code_length", binary_code_length)
    configs['binary_code_length'] = binary_code_length

    net = BinaryCodeNet_Deeplab(
        num_resnet_layers=resnet_layer,
        concat=concat,
        binary_code_length=binary_code_length,
        divided_number_each_iteration=divide_number_each_itration,
        output_kernel_size=output_kernel_size,
        efficientnet_key=efficientnet_key
    )
    if torch.cuda.is_available():
        net = net.cuda()

    checkpoint = torch.load(configs['checkpoint_file'])
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    fig = plt.figure()
    ani = FuncAnimation(fig, process, interval=33)  # Aggiorna il plot ogni secondo
    plt.show()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str)  # config file
    parser.add_argument('--obj_name', type=str)
    parser.add_argument('--ckpt_file', type=str)
    parser.add_argument('--ignore_bit', type=str)
    parser.add_argument('--eval_output_path', type=str)
    parser.add_argument('--use_icp', type=str, choices=('True', 'False'), default='False')  # config file
    args = parser.parse_args()
    config_file = args.cfg
    checkpoint_file = args.ckpt_file
    eval_output_path = args.eval_output_path
    obj_name = args.obj_name
    configs = parse_cfg(config_file)

    configs['use_icp'] = (args.use_icp == 'True')
    configs['obj_name'] = obj_name

    if configs['Detection_reaults'] != 'none':
        Detection_reaults = configs['Detection_reaults']
        dirname = os.path.dirname(__file__)
        Detection_reaults = os.path.join(dirname, Detection_reaults)
        configs['Detection_reaults'] = Detection_reaults

    configs['checkpoint_file'] = checkpoint_file
    configs['eval_output_path'] = eval_output_path

    configs['ignore_bit'] = int(args.ignore_bit)

    # print the configurations
    for key in configs:
        print(key, " : ", configs[key], flush=True)

    main(configs)
