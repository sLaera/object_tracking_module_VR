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

import socket

import json
import base64

import traceback

from ultralytics import YOLO

HOST = '127.0.0.1'
PORT = 6543


# freeze the given number of layers of the given model
def freeze(model, n=None):
    """
    this function freezes the given number of layers of the given model in order to fine-tune it.
    :param model: the model to be frozen
    :param n: the number of layers to be frozen, if None, all the layers are frozen
    :return: the frozen model
    """
    if n is None:
        for param in model.parameters():
            param.requires_grad = False
    else:
        count = 0
        for param in model.parameters():
            if count < n:
                param.requires_grad = False
            count += 1


def count_parameters(model, count_embedding=True):
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += 1
    print("Total number of parameters: {}.".format(total_params))
    return total_params


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


def add_box_to_image(image, bbox, conf):
    position = (10, 30)  # Coordinate (x, y) della posizione del testo

    # Imposta i parametri della scritta
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 0, 0)  # BGR (Blu, Verde, Rosso)
    thickness = 2
    line_type = cv2.LINE_AA
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    image = cv2.flip(image, 1)

    # Aggiungi la scritta all'immagine
    cv2.putText(image, f"{conf:.2f}", position, font, font_scale, font_color, thickness, line_type)

    return image


def main(configs):
    def process(_b_box):

        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            return

        # --------------------predict------------------------------------------------------------------
        # test_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        test_img = frame
        # black and white image
        '''test_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        result_image = cv2.merge((gray_image, gray_image, gray_image))
        test_img = result_image'''

        '''# per test questi bbox sono falsi
        b_box = [200, 40, 140, 140]
        #test_img = cv2.cvtColor(cv2.imread("1.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)'''

        # find bbox
        yolo_results = yolo.predict(test_img, device=0, max_det=10, conf=0.4)  # return a list of Results objects
        conf = 0
        if yolo_results[0].boxes:
            box = max(yolo_results[0].boxes, key=lambda b: b.conf)
            x, y, w, h = box.xywh.cpu().numpy()[0]
            # convert x,y from center to left high position of bbox
            x = x - w / 2
            y = y - h / 2
            _b_box = np.array([x, y, w, h]).astype(int)
            conf = box.conf.cpu().numpy()[0]

        roi_x = get_roi(test_img, _b_box, BoundingBox_CropSize_image, interpolation=cv2.INTER_LINEAR,
                        resize_method=resize_method)

        test_img = add_box_to_image(test_img, _b_box, conf)

        # show_image("test", roi_x.copy())
        cv2.imshow('camera', test_img)

        if no_socket == "True":
            return _b_box

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
        cv2.imshow('mask', pred_masks[0] * 255)

        # show_image("pred mask", pred_masks)
        R = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        T = np.zeros([1, 3])

        R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[0],
                                                                   pred_code_images[0],
                                                                   _b_box, BoundingBox_CropSize_GT,
                                                                   divide_number_each_itration,
                                                                   dict_class_id_3D_points,
                                                                   intrinsic_matrix=cam_K)

        if success:
            R = R_predict
            T = t_predict.flatten()
            # print(R_predict, t_predict)
            # image_with_indicator = draw_indicator(frame, R_predict, t_predict.flatten())
            # ------------------------------------------------------------------------------------------------
            # inviare dati
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R
            transformation_matrix[:3, 3] = T * 0.01
            mat_json = {"matrix": transformation_matrix.tolist()}
            mat_json = {
                "matrix": {
                    "r0": transformation_matrix[0, :].tolist(),
                    "r1": transformation_matrix[1, :].tolist(),
                    "r2": transformation_matrix[2, :].tolist(),
                    "r3": transformation_matrix[3, :].tolist()
                }
            }
            mat_json = json.dumps(mat_json)
            conn.sendall(mat_json.encode('utf-8'))

        return _b_box
        # update_plot(R, T.flatten(), test_img)  # T.flatten()

    # ---------- -------- ---------- ---------- ----------- ----------- ---------
    bop_path = configs['bop_path']
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    BoundingBox_CropSize_image = configs['BoundingBox_CropSize_image']  # input image size
    BoundingBox_CropSize_GT = configs['BoundingBox_CropSize_GT']  # network output size
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']  # now only support "L1" or "BCE"
    output_kernel_size = configs['output_kernel_size']  # last layer kernel size
    resnet_layer = configs['resnet_layer']  # usually resnet 34
    concat = configs['concat_encoder_decoder']
    efficientnet_key = configs['efficientnet_key']
    resize_method = configs['resize_method']
    yolo_pt = configs['yolo_pt']
    no_socket = configs['no_socket']

    # pixel code settings
    divide_number_each_itration = configs['divide_number_each_itration']
    number_of_itration = configs['number_of_itration']

    # todo: use the camera calibration matrix get by a real camera
    # cam_K = cam_params[obj_id][0]['cam_K'].reshape((3, 3))
    cam_K = [[1.50557939e+03, 0.00000000e+00, 2.91158435e+02],
             [0.00000000e+00, 1.47186344e+03, 1.02544916e+02],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

    obj_name_obj_id, symmetry_obj = get_obj_info(dataset_name)
    obj_id = int(obj_name_obj_id[obj_name] - 1)  # now the obj_id started from 0

    dataset_dir = os.path.join(bop_path, dataset_name)
    path_dict = os.path.join(dataset_dir, "models_GT_color", "Class_CorresPoint{:06d}.txt".format(obj_id + 1))
    total_numer_class, _, _, dict_class_id_3D_points = load_dict_class_id_3D_points(path_dict)
    divide_number_each_itration = int(divide_number_each_itration)
    total_numer_class = int(total_numer_class)
    number_of_itration = int(number_of_itration)
    if divide_number_each_itration ** number_of_itration != total_numer_class:
        raise AssertionError("the combination is not valid")

    if divide_number_each_itration != 2 and (BinaryCode_Loss_Type == 'BCE' or BinaryCode_Loss_Type == 'L1'):
        raise AssertionError("for non-binary case, use CE as loss function")
    if divide_number_each_itration == 2 and BinaryCode_Loss_Type == 'CE':
        raise AssertionError("not support for now")

    torch.manual_seed(0)  # the both are only good for ablation study
    np.random.seed(0)  # if can be removed in the final experiments

    binary_code_length = number_of_itration
    print("predicted binary_code_length", binary_code_length)
    configs['binary_code_length'] = binary_code_length

    # ------YOLO MODEL ---------
    yolo = YOLO(yolo_pt)  # pretrained YOLOv8n model

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

    checkpoint = torch.load(configs['checkpoint_file'], map_location=torch.device('cuda:0'))
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    b_box = [1, 1, 1, 1]
    if no_socket and no_socket == "True":
        while True:
            b_box = process(b_box)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Associa il socket all'indirizzo IP e alla porta definiti
                s.bind((HOST, PORT))
                # Metti il server in ascolto per una connessione in entrata
                s.listen()
                print("Server in ascolto...")
                conn, addr = s.accept()
                print('Connesso da:', addr)
                while True:
                    b_box = process(b_box)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception:
                print(traceback.format_exc())
            finally:
                s.close()

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str)  # config file
    parser.add_argument('--obj_name', type=str)
    parser.add_argument('--ckpt_file', type=str)
    parser.add_argument('--yolo_pt', type=str)  # path to the trained yolo checkpoint 
    parser.add_argument('--no_socket', type=str)
    args = parser.parse_args()
    config_file = args.cfg
    checkpoint_file = args.ckpt_file
    obj_name = args.obj_name
    configs = parse_cfg(config_file)

    configs['obj_name'] = obj_name
    configs['checkpoint_file'] = checkpoint_file
    configs['yolo_pt'] = args.yolo_pt
    configs['no_socket'] = args.no_socket

    # print the configurations
    for key in configs:
        print(key, " : ", configs[key], flush=True)

    main(configs)
