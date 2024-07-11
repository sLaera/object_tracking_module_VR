import open3d as o3d
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv
import cv2
import os

def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def set_model_pose(R_m2c, t_m2c):
    R_m2c = np.array(R_m2c).reshape(3, 3)
    t_m2c = np.array(t_m2c)
    # Costruisci la matrice di trasformazione completa
    model_matrix_camera = np.eye(4)
    model_matrix_camera[:3, :3] = R_m2c.T
    model_matrix_camera[:3, 3] = t_m2c

    # model_matrix_world = cam_matrix_world @ model_matrix_camera
    model_matrix_world = model_matrix_camera

    return model_matrix_world


#--TO EDIT WITH THE CORRECT FOLDER
folder = "..\\BOP ROOT LIGHT\\ycbv_light\\train_real_recreated"
obj_id = 18
ply_name = "obj_000018.ply"
IMAGE_W = 640 # 600
IMAGE_H = 480 # 450
INITIAL_ROTATION = [180,0,0] #[180,0,0]
R_bcam2cv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) #  np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# Percorsi ai file JSON
scene_camera_filepath = f"{folder}\\000000\\scene_camera.json"
scene_gt_filepath = f"{folder}\\000000\\scene_gt.json"
ply_file_path = f"models\\{ply_name}"

#make the folders
os.makedirs(f"{folder}_GT", exist_ok=True)
os.makedirs(f"{folder}_GT\\000000", exist_ok=True)
basepath = f"{folder}_GT\\000000"

# Carica i dati JSON
scene_camera_data = load_json(scene_camera_filepath)
scene_gt_data = load_json(scene_gt_filepath)

# Imposta la visualizzazione
vis = o3d.visualization.Visualizer()
vis.create_window(width=1920, height=1057, visible=True)

mesh = o3d.io.read_triangle_mesh(ply_file_path)
vis.add_geometry(mesh)

for key, cam_data in scene_camera_data.items():
    print(key)
    
    # Estrai i dati della camera
    cam_data = scene_camera_data[key]
    cam_R_w2c = cam_data["cam_R_w2c"]
    cam_t_w2c = cam_data["cam_t_w2c"]

    # Estrai i dati del modello
    model_data = [item for item in scene_gt_data[key] if item["obj_id"] == obj_id][0]
    cam_R_m2c = model_data["cam_R_m2c"]
    cam_t_m2c = model_data["cam_t_m2c"]
    
        # Imposta la posizione della camera
    camera_intrinsics = np.array(cam_data["cam_K"]).reshape(3, 3)

    model_pose = set_model_pose(
        model_data["cam_R_m2c"], model_data["cam_t_m2c"]
    )

    #adjust initial rotation
    rotation = R.from_euler("xyz", INITIAL_ROTATION, degrees=True)
    rot_matrix = rotation.as_matrix()
    transform = np.hstack((rot_matrix, np.array([0, 0, 0]).reshape(3, 1)))
    transform = np.vstack((transform, [0, 0, 0, 1]))

    
    model_pose = R_bcam2cv @ model_pose

    mesh_transform = model_pose @ transform
    mesh.transform(mesh_transform)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        IMAGE_W,
        IMAGE_H,
        camera_intrinsics[0, 0],
        camera_intrinsics[1, 1],
        camera_intrinsics[0, 2],
        camera_intrinsics[1, 2],
    )
    intrinsic.intrinsic_matrix = camera_intrinsics
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic
    
    cam.extrinsic = np.eye(4, 4)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    print(np.abs(mesh_transform[2][3]))
    ctr.set_constant_z_far(np.abs(mesh_transform[2][3]) + 10000)
    ctr.set_zoom(1)
    
    # Disabilita luci e ombre
    opt = vis.get_render_option()
    opt.background_color = [0, 0, 0]
    opt.light_on = False  # Disabilita tutte le luci

    #render
    scene_id = '{:06d}'.format(int(key))
    obj_id_str = '{:06d}'.format(int(obj_id))
    output_path = basepath + f"\\{scene_id}_{obj_id_str}.png"

    # Avvia la visualizzazione per aggiornare la scena
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path)
    
    inverse_transformation = np.linalg.inv(mesh_transform)
    mesh.transform(inverse_transformation)

    '''#the render must be at a higher resolution, crop to fit 600x640
    image = cv2.imread(output_path)

    # Dimensioni originali dell'immagine
    original_height, original_width = image.shape[:2]

    # Ridimensiona l'immagine al 89%
    scale_percent = 89
    width = int(original_width * scale_percent / 100)
    height = int(original_height * scale_percent / 100)
    dim = (width, height)

    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Dimensioni dell'immagine ridimensionata
    resized_height, resized_width = resized_image.shape[:2]

    # Dimensioni del ritaglio
    crop_width = IMAGE_W
    crop_height = IMAGE_H

    # Calcola le coordinate per ritagliare il centro
    start_x = resized_width // 2 - crop_width // 2
    start_y = resized_height // 2 - crop_height // 2

    # Effettua il ritaglio
    cropped_image = resized_image[start_y:start_y + crop_height, start_x:start_x + crop_width]

    # Salva l'immagine risultante
    cv2.imwrite(output_path, cropped_image)'''
    
    while True:
        vis.poll_events()
        vis.update_renderer()

vis.destroy_window()





