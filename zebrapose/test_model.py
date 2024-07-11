import os
import pdb
import sys

import cv2

sys.path.insert(0, os.getcwd())

from config_parser import parse_cfg
import argparse

from tools_for_BOP import bop_io
from tools_for_BOP.common_dataset_info import get_obj_info
from bop_dataset_pytorch import bop_dataset_single_obj_pytorch

import torch
from torch import optim
import numpy as np

from binary_code_helper.CNN_output_to_pose import load_dict_class_id_3D_points, CNN_outputs_to_object_pose

sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout
from model.BinaryCodeNet import BinaryCodeNet_Deeplab
from model.BinaryCodeNet import MaskLoss, BinaryCodeLoss

from torch.utils.tensorboard import SummaryWriter

from utils import save_checkpoint, get_checkpoint, save_best_checkpoint
from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP

from get_detection_results import get_detection_results, ycbv_select_keyframe

from common_ops import from_output_to_class_mask, from_output_to_class_binary_code, get_batch_size

from test_network_with_test_data import test_network_with_single_obj
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# freeze the given number of layers of the given model
# freeze the given number of layers of the given model
def freeze_layers(model, n=None):
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


def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += 1
    print("Total number of parameters: {}.".format(total_params))
    return total_params

def invert_class_id_image(class_id_image):
    B = np.right_shift(class_id_image, 16) & 0xFF
    G = np.right_shift(class_id_image, 8) & 0xFF
    R = class_id_image & 0xFF
    BGR_image = np.stack((B, G, R), axis=-1)
    BGR_image = BGR_image.astype(np.uint8)
    return BGR_image

def normalize_array(arr, min_val, max_val):
    # Ensure the array is a numpy array
    arr = np.array(arr, dtype=float)
    
    # Map the array to the range [0, 1]
    normalized_arr = (arr - min_val) / (max_val - min_val)
    
    return normalized_arr

def mean_square_error(true, pred):
    return np.mean((true - pred) ** 2)

def main(configs):
    print("______________________get config____________________________")
    config_file_name = configs['config_file_name']
    #### training dataset
    bop_challange = configs['bop_challange']
    bop_path = configs['bop_path']
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    training_data_folder = configs['training_data_folder']
    training_data_folder_2 = configs['training_data_folder_2']
    val_folder = configs['val_folder']  # usually is 'test'
    second_dataset_ratio = configs['second_dataset_ratio']  # the percentage of second dataset in the batch
    num_workers = configs['num_workers']  # for data loader
    train_obj_visible_theshold = configs[
        'train_obj_visible_theshold']  # for test is always 0.1, for training we can set different values
    #### network settings
    BoundingBox_CropSize_image = configs['BoundingBox_CropSize_image']  # input image size
    BoundingBox_CropSize_GT = configs['BoundingBox_CropSize_GT']  # network output size
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']  # now only support "L1" or "BCE"
    mask_binary_code_loss = configs['mask_binary_code_loss']  # if binary code loss only applied for object mask region
    use_histgramm_weighted_binary_loss = configs['use_histgramm_weighted_binary_loss']
    output_kernel_size = configs['output_kernel_size']  # last layer kernel size
    resnet_layer = configs['resnet_layer']  # usually resnet 34
    concat = configs['concat_encoder_decoder']
    predict_entire_mask = configs[
        'predict_entire_mask']  # if predict the entire object part rather than the visible one
    if 'efficientnet_key' in configs.keys():
        efficientnet_key = configs['efficientnet_key']
    #### check points
    load_checkpoint = configs['load_checkpoint']
    tensorboard_path = configs['tensorboard_path']
    check_point_path = configs['check_point_path']
    total_iteration = configs['total_iteration']  # train how many steps
    #### transfer learning
    if 'load_pretrained' in configs:
        load_pretrained = configs['load_pretrained']
    else:
        load_pretrained = False
    if 'pretrained_path' in configs:
        pretrained_path = configs['pretrained_path']
    else:
        pretrained_path = 'None'
    if 'freeze' in configs:
        freeze = configs['freeze']
    else:
        freeze = False
    if 'n_layer_multyplayer' in configs:
        n_layer_multyplayer = configs['n_layer_multyplayer']
    else:
        n_layer_multyplayer = 0
    ####split
    if 'split_train_dataset' in configs:
        split_train_dataset = configs['split_train_dataset']
    else:
        split_train_dataset = False
    if 'split_ratio_train' in configs:
        split_ratio_train = configs['split_ratio_train']
    else:
        split_ratio_train = 1
    #### optimizer
    optimizer_type = configs['optimizer_type']  # Adam is the best sofar
    batch_size = configs['batch_size']  # 32 is the best so far, set to 16 for debug in local machine
    learning_rate = configs['learning_rate']  # 0.002 or 0.003 is the best so far
    binary_loss_weight = configs['binary_loss_weight']  # 3 is the best so far
    #### augmentations
    Detection_reaults = configs['Detection_reaults']  # for the test, the detected bounding box provided by GDR Net
    padding_ratio = configs['padding_ratio']  # pad the bounding box for training and test
    resize_method = configs['resize_method']  # how to resize the roi images to 256*256
    use_peper_salt = configs['use_peper_salt']  # if add additional peper_salt in the augmentation
    use_motion_blur = configs['use_motion_blur']  # if add additional motion_blur in the augmentation
    # vertex code settings
    divide_number_each_itration = configs['divide_number_each_itration']
    number_of_itration = configs['number_of_itration']

    sym_aware_training = configs['sym_aware_training']

    # get dataset informations
    print("______________________get dataset information__________________________")
    dataset_dir, source_dir, model_plys, model_info, model_ids, rgb_files, depth_files, mask_files, mask_visib_files, gts, gt_infos, cam_param_global, cam_params = bop_io.get_dataset(
        bop_path, dataset_name, train=True, data_folder=training_data_folder, data_per_obj=True, incl_param=True,
        train_obj_visible_theshold=train_obj_visible_theshold)
    obj_name_obj_id, symmetry_obj = get_obj_info(dataset_name)
    obj_id = int(obj_name_obj_id[obj_name] - 1)  # now the obj_id started from 0
    if obj_name in symmetry_obj:
        Calculate_Pose_Error = Calculate_ADI_Error_BOP
    else:
        Calculate_Pose_Error = Calculate_ADD_Error_BOP
    mesh_path = model_plys[obj_id + 1]  # mesh_path is a dict, the obj_id should start from 1
    print(mesh_path, flush=True)
    obj_diameter = model_info[str(obj_id + 1)]['diameter']
    print("obj_diameter", obj_diameter, flush=True)
    path_dict = os.path.join(dataset_dir, "models_GT_color", "Class_CorresPoint{:06d}.txt".format(obj_id + 1))
    total_numer_class, _, _, dict_class_id_3D_points = load_dict_class_id_3D_points(path_dict)
    divide_number_each_itration = int(divide_number_each_itration)
    total_numer_class = int(total_numer_class)
    number_of_itration = int(number_of_itration)
    if divide_number_each_itration ** number_of_itration != total_numer_class:
        raise AssertionError("the combination is not valid")
    if BoundingBox_CropSize_image / BoundingBox_CropSize_GT != 2:
        raise AssertionError("currnet endoder-decoder only support input_size/output_size = 2")
    GT_code_infos = [divide_number_each_itration, number_of_itration, total_numer_class]

    if divide_number_each_itration != 2 and (BinaryCode_Loss_Type == 'BCE' or BinaryCode_Loss_Type == 'L1'):
        raise AssertionError("for non-binary case, use CE as loss function")
    if divide_number_each_itration == 2 and BinaryCode_Loss_Type == 'CE':
        raise AssertionError("not support for now")

    vertices = inout.load_ply(mesh_path)["pts"]

    ########################## define data loader
    print("______________________define data loader__________________________")
    batch_size_1_dataset, batch_size_2_dataset = get_batch_size(second_dataset_ratio, batch_size)
    batch_size_test = batch_size
    if training_data_folder_2 != 'none':
            batch_size_test = batch_size_1_dataset
    batch_size_test = int(batch_size_test/2)
    if batch_size_test < 1:
        batch_size_test = 1

    batch_size = batch_size_test
    # define test data loader
    if not bop_challange:
        dataset_dir_test, _, _, _, _, test_rgb_files, _, test_mask_files, test_mask_visib_files, test_gts, test_gt_infos, _, camera_params_test = bop_io.get_dataset(
            bop_path, dataset_name, train=False, data_folder=val_folder, data_per_obj=True, incl_param=True,
            train_obj_visible_theshold=train_obj_visible_theshold)
        if dataset_name == 'ycbv' and Detection_reaults != 'none':
            print("select key frames from ycbv test images")
            key_frame_index = ycbv_select_keyframe(Detection_reaults, test_rgb_files[obj_id])
            test_rgb_files_keyframe = [test_rgb_files[obj_id][i] for i in key_frame_index]
            test_mask_files_keyframe = [test_mask_files[obj_id][i] for i in key_frame_index]
            test_mask_visib_files_keyframe = [test_mask_visib_files[obj_id][i] for i in key_frame_index]
            test_gts_keyframe = [test_gts[obj_id][i] for i in key_frame_index]
            test_gt_infos_keyframe = [test_gt_infos[obj_id][i] for i in key_frame_index]
            camera_params_test_keyframe = [camera_params_test[obj_id][i] for i in key_frame_index]
            test_rgb_files[obj_id] = test_rgb_files_keyframe
            test_mask_files[obj_id] = test_mask_files_keyframe
            test_mask_visib_files[obj_id] = test_mask_visib_files_keyframe
            test_gts[obj_id] = test_gts_keyframe
            test_gt_infos[obj_id] = test_gt_infos_keyframe
            camera_params_test[obj_id] = camera_params_test_keyframe
    else:
        dataset_dir_test, _, _, _, _, test_rgb_files, _, test_mask_files, test_mask_visib_files, test_gts, test_gt_infos, _, camera_params_test = bop_io.get_bop_challange_test_data(
            bop_path, dataset_name, target_obj_id=obj_id + 1, data_folder=val_folder)
    print('test_rgb_file exsample', test_rgb_files[obj_id][0])

    if Detection_reaults != 'none':
        Det_Bbox = get_detection_results(Detection_reaults, test_rgb_files[obj_id], obj_id + 1, 0)
    else:
        Det_Bbox = None

    test_dataset = bop_dataset_single_obj_pytorch(
        dataset_dir_test, val_folder, test_rgb_files[obj_id], test_mask_files[obj_id], test_mask_visib_files[obj_id],
        test_gts[obj_id], test_gt_infos[obj_id], camera_params_test[obj_id], False,
        BoundingBox_CropSize_image, BoundingBox_CropSize_GT, GT_code_infos,
        padding_ratio=padding_ratio, resize_method=resize_method, Detect_Bbox=Det_Bbox,
        use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur, sym_aware_training=sym_aware_training
    )

    print("number of test images: ", len(test_dataset), flush=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False,
                                            num_workers=num_workers)
    #############build the network 
    print("______________________build the network __________________________")
    binary_code_length = number_of_itration
    print("binary_code_length: ", binary_code_length)
    configs['binary_code_length'] = binary_code_length

    net = BinaryCodeNet_Deeplab(
        num_resnet_layers=resnet_layer,
        concat=concat,
        binary_code_length=binary_code_length,
        divided_number_each_iteration=divide_number_each_itration,
        output_kernel_size=output_kernel_size,
        efficientnet_key=efficientnet_key
    )

    # visulize input image, ground truth code, ground truth mask
    writer = SummaryWriter(tensorboard_path + "_TESTDATA_2")

    # visulize_input_data_and_network(writer, train_loader, net)

    if torch.cuda.is_available():
        net = net.cuda()

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        lr = optimizer.param_groups[0]['lr']
        betas = optimizer.param_groups[0]['betas']
        eps = optimizer.param_groups[0]['eps']
        weight_decay = optimizer.param_groups[0]['weight_decay']
        print(lr,betas,eps,weight_decay)
    else:
        raise NotImplementedError(f"unknown optimizer type: {optimizer_type}")

    best_score_path = os.path.join(check_point_path, 'best_score')
    if not os.path.isdir(best_score_path):
        os.makedirs(best_score_path)
    best_score = 0
    iteration_step = 0
    if load_checkpoint:
        checkpoint = torch.load(get_checkpoint(check_point_path))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_score = checkpoint['best_score']
        iteration_step = checkpoint['iteration_step']
    elif load_pretrained:
        checkpoint = torch.load(pretrained_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if freeze:
        num_param_to_freeze = count_parameters(net) * float(n_layer_multyplayer)
        freeze_layers(net, int(num_param_to_freeze))
        

    print("______________________Start Testing__________________________")
    net.eval()
    for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, class_code_images, cam_Ks) in enumerate(
                tqdm(test_loader)):
        print(f"______________________{batch_idx}__________________________")
        # data to GPU
        if torch.cuda.is_available():
            data = data.cuda()
            entire_masks = entire_masks.cuda()
            masks = masks.cuda()
            class_code_images = class_code_images.cuda()

        if data.shape[0] != batch_size:
            raise ValueError(f"batch size wrong")
        pred_mask_prob, pred_code_prob = net(data)

        pred_masks = from_output_to_class_mask(pred_mask_prob)
        pred_codes = from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type,
                                                        divided_num_each_interation=divide_number_each_itration,
                                                        binary_code_length=binary_code_length)

        pred_codes = pred_codes.transpose(0, 2, 3, 1)

        pred_masks = pred_masks.transpose(0, 2, 3, 1)
        pred_masks = pred_masks.squeeze(axis=-1)

        Rs = Rs.detach().cpu().numpy()
        ts = ts.detach().cpu().numpy()
        Bboxes = Bboxes.detach().cpu().numpy()
        cam_Ks = cam_Ks.detach().cpu().numpy()

        ADD_passed = np.zeros(batch_size)
        ADD_error = np.zeros(batch_size)

        ERROR_GT_R = np.zeros(batch_size)
        ERROR_GT_T = np.zeros(batch_size)
        ERROR_GT_T_X = np.zeros(batch_size)
        ERROR_GT_T_Y = np.zeros(batch_size)
        ERROR_GT_T_Z = np.zeros(batch_size)

        ERROR_R = np.zeros(batch_size)
        ERROR_T = np.zeros(batch_size)
        ERROR_T_X = np.zeros(batch_size)
        ERROR_T_Y = np.zeros(batch_size)
        ERROR_T_Z = np.zeros(batch_size)

        n_errors = 0
        n_errors_GT = 0
        for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
            R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter],
                                                                        pred_codes[counter],
                                                                        Bbox,
                                                                        BoundingBox_CropSize_GT,
                                                                        divide_number_each_itration,
                                                                        dict_class_id_3D_points,
                                                                        intrinsic_matrix=cam_K)

            R_GT_calculated, t_GT_calculated, success_GT = CNN_outputs_to_object_pose(masks[counter].detach().cpu().numpy().astype(np.uint8), np.transpose(class_code_images[counter].detach().cpu().numpy().astype(np.uint8), (1, 2, 0)), 
                                                                    Bbox, BoundingBox_CropSize_GT, divide_number_each_itration, dict_class_id_3D_points, 
                                                                    intrinsic_matrix=cam_K)
            
            
            

            #Test matrix transform
            GT_matrix = np.hstack((r_GT, t_GT))
            GT_matrix = np.vstack((GT_matrix,[0,0,0,1]))
            scale = np.eye(4,4) * (1/0.022334)
            GT_matrix = scale @ GT_matrix
            GT_matrix[2,3] *= -1
            r_GT = GT_matrix[:3, :3]
            t_GT =  GT_matrix[:3, 3]
            t_GT = t_GT.reshape(-1, 1)
            
            if not success_GT:
                n_errors_GT += 1
            if not success:
                n_errors += 1
                continue
                

            if success_GT and success:
                ERROR_GT_R[counter] = mean_square_error(R_GT_calculated, R_predict)
                ERROR_GT_T[counter] = mean_square_error(t_GT_calculated, t_predict)
                ERROR_GT_T_X[counter] = (t_GT_calculated[0,0] - t_predict[0,0]) ** 2
                ERROR_GT_T_Y[counter] = (t_GT_calculated[1,0] - t_predict[1,0]) ** 2
                ERROR_GT_T_Z[counter] = (t_GT_calculated[2,0] - t_predict[2,0]) ** 2

                ERROR_R[counter] = mean_square_error(r_GT, R_predict)
                ERROR_T[counter] = mean_square_error(t_GT, t_predict)
                ERROR_T_X[counter] = (t_GT[0,0] - t_predict[0,0]) ** 2
                ERROR_T_Y[counter] = (t_GT[1,0] - t_predict[1,0]) ** 2
                ERROR_T_Z[counter] = (t_GT[2,0] - t_predict[2,0]) ** 2
            else:
                ERROR_GT_R[counter] = 10000
                ERROR_GT_T[counter] = 10000
                ERROR_GT_T_X[counter] = 10000
                ERROR_GT_T_Y[counter] = 10000
                ERROR_GT_T_Z[counter] = 10000

                ERROR_R[counter] = 10000
                ERROR_T[counter] = 10000
                ERROR_T_X[counter] = 10000
                ERROR_T_Y[counter] = 10000
                ERROR_T_Z[counter] = 10000

            add_error = 10000
            if success:
                add_error = Calculate_Pose_Error(r_GT, t_GT, R_predict, t_predict, vertices)
                if np.isnan(add_error):
                    add_error = 10000
            if add_error < obj_diameter * 0.1:
                ADD_passed[counter] = 1
            ADD_error[counter] = add_error

        ADD_passed = np.mean(ADD_passed)
        ADD_error = np.mean(ADD_error)

        writer.add_scalar('ADD/ADD_passed', ADD_passed, iteration_step)
        writer.add_scalar('ADD/ADD_Error', ADD_error, iteration_step)

        ERROR_GT_R = np.sqrt(np.mean(ERROR_GT_R))
        ERROR_GT_T = np.sqrt(np.mean(ERROR_GT_T))
        ERROR_GT_T_X = np.sqrt(np.mean(ERROR_GT_T_X))
        ERROR_GT_T_Y = np.sqrt(np.mean(ERROR_GT_T_Y))
        ERROR_GT_T_Z = np.sqrt(np.mean(ERROR_GT_T_Z))

        ERROR_R = np.sqrt(np.mean(ERROR_R))
        ERROR_T = np.sqrt(np.mean(ERROR_T))
        ERROR_T_X = np.sqrt(np.mean(ERROR_T_X))
        ERROR_T_Y = np.sqrt(np.mean(ERROR_T_Y))
        ERROR_T_Z = np.sqrt(np.mean(ERROR_T_Z))

        writer.add_scalar('ERRORS/pyprogressivex/r_RMSE', ERROR_GT_R, iteration_step)
        writer.add_scalar('ERRORS/pyprogressivex/t_RMSE', ERROR_GT_T, iteration_step)
        writer.add_scalar('ERRORS/pyprogressivex/t_x_RMSE', ERROR_GT_T_X, iteration_step)
        writer.add_scalar('ERRORS/pyprogressivex/t_y_RMSE', ERROR_GT_T_Y, iteration_step)
        writer.add_scalar('ERRORS/pyprogressivex/t_z_RMSE', ERROR_GT_T_Z, iteration_step)

        writer.add_scalar('ERRORS/GT/r_RMSE', ERROR_R, iteration_step)
        writer.add_scalar('ERRORS/GT/t_RMSE', ERROR_T, iteration_step)
        writer.add_scalar('ERRORS/GT/t_x_RMSE', ERROR_T_X, iteration_step)
        writer.add_scalar('ERRORS/GT/t_y_RMSE', ERROR_T_Y, iteration_step)
        writer.add_scalar('ERRORS/GT/t_z_RMSE', ERROR_T_Z, iteration_step)

        writer.add_scalar('ERRORS/nerrorsGT', n_errors_GT, iteration_step)
        writer.add_scalar('ERRORS/nerrors', n_errors, iteration_step)

        # stop the training erlyaer
        iteration_step = iteration_step + 1
        if iteration_step >= total_iteration:
            break    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str)  # config file
    parser.add_argument('--obj_name', type=str)  # config file
    parser.add_argument('--sym_aware_training', type=str, choices=('True', 'False'), default='False')  # config file
    args = parser.parse_args()
    config_file = args.cfg
    configs = parse_cfg(config_file)
    configs['obj_name'] = args.obj_name
    configs['sym_aware_training'] = (args.sym_aware_training == 'True')

    check_point_path = configs['check_point_path']
    tensorboard_path = configs['tensorboard_path']

    config_file_name = os.path.basename(config_file)
    config_file_name = os.path.splitext(config_file_name)[0]
    check_point_path = check_point_path + config_file_name
    tensorboard_path = tensorboard_path + config_file_name
    configs['check_point_path'] = check_point_path + args.obj_name + '/'
    configs['tensorboard_path'] = tensorboard_path + args.obj_name + '/'

    configs['config_file_name'] = config_file_name

    if configs['Detection_reaults'] != 'none':
        Detection_reaults = configs['Detection_reaults']
        dirname = os.path.dirname(__file__)
        Detection_reaults = os.path.join(dirname, Detection_reaults)
        configs['Detection_reaults'] = Detection_reaults

    # print the configurations
    for key in configs:
        print(key, " : ", configs[key], flush=True)

    if 'device_cuda_number' in configs:
        device_cuda_number = configs['device_cuda_number']
    else:
        device_cuda_number = 0

    with torch.cuda.device(device_cuda_number):
        main(configs)
