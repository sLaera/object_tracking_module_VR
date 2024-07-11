import open3d as o3d
import json
import os
import numpy as np
from numpy.linalg import inv
import cv2 as cv

# Funzione per caricare i dati JSON da un file
def load_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def set_camera_pose(vis, R_w2c, t_w2c):
    # Convert lists to numpy arrays
    R_w2c = np.array(R_w2c).reshape(3, 3)
    t_w2c = np.array(t_w2c)
    '''
    R_bcam2cv = np.array([[1, 0,  0],
                          [0, -1, 0],
                          [0, 0, -1]])
    
    R_bcam2cv_inv = inv(R_bcam2cv)
    # Calcola R_world2bcam
    R_world2bcam = R_bcam2cv_inv @ R_w2c
    R_world2bcam = R_world2bcam.T

    # Calcola T_world2bcam
    T_world2bcam = R_bcam2cv_inv @ t_w2c 

    # Ora combina R_world2bcam e T_world2bcam per ottenere la matrice di trasformazione della camera
    cam_matrix_world = np.eye(4)
    cam_matrix_world[:3, :3] = R_world2bcam
    cam_matrix_world[:3, 3] = T_world2bcam'''
    
    cam_matrix_world = np.eye(4)
    cam_matrix_world[:3, :3] = R_w2c
    cam_matrix_world[:3, 3] = t_w2c
    
    # Imposta la posizione e la rotazione della camera
    vis.get_view_control().set_lookat(cam_matrix_world[:3, 3])
    vis.get_view_control().set_up(cam_matrix_world[:3, 1])
    return cam_matrix_world

def set_model_pose(mesh, R_m2c, t_m2c, cam_matrix_world):
    R_m2c = np.array(R_m2c).reshape(3, 3)
    t_m2c = np.array(t_m2c)
    # Costruisci la matrice di trasformazione completa
    model_matrix_camera = np.eye(4)
    model_matrix_camera[:3, :3] = R_m2c.T
    model_matrix_camera[:3, 3] = t_m2c
    
    model_matrix_world = cam_matrix_world @ model_matrix_camera
    
    mesh.transform(model_matrix_world)
    vis.update_geometry(mesh)

def calculate_fov(focal_length_mm, sensor_size_mm):
    fov_rad = 2 * np.arctan(sensor_size_mm / (2 * focal_length_mm))
    fov_deg = np.degrees(fov_rad)
    return fov_deg

def init_renderer(ply_file_path):
    # Crea una finestra di visualizzazione
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=600, height=450,visible=True)
    # Disabilita luci e ombre
    opt = vis.get_render_option()
    opt.background_color = [0, 0, 0] 
    opt.light_on = False  # Disabilita tutte le luci
    #opt.mesh_shade_option = o3d.visualization.RenderOption.ShadeOption.FLAT  # Utilizza ombreggiatura flat (senza ombre)

    # Imposta la lunghezza focale della camera
    fov_step = calculate_fov(35, 600)
    vis.get_view_control().change_field_of_view(step=fov_step)
    
    # Carica il file PLY
    mesh = o3d.io.read_triangle_mesh(ply_file_path)
    vis.add_geometry(mesh)
    return mesh, vis

def render_ply_to_image(vis, output_image_path):
    
    # Avvia la visualizzazione per aggiornare la scena
    vis.poll_events()
    vis.update_renderer()

    # Salva il rendering come immagine
    #vis.capture_screen_image(output_image_path)

    
    

#--TO EDIT WITH THE CORRECT FOLDER
folder = "datasets\\train"

# Percorsi ai file JSON
scene_camera_filepath = f"{folder}\\000000\\scene_camera.json"
scene_gt_filepath = f"{folder}\\000000\\scene_gt.json"
ply_file_path = f"models\\GT.ply"

#make the folders
os.makedirs(f"{folder}_GT", exist_ok=True)
os.makedirs(f"{folder}_GT\\000000", exist_ok=True)
basepath = f"{folder}_GT\\000000"

# Carica i dati JSON
scene_camera_data = load_json(scene_camera_filepath)
scene_gt_data = load_json(scene_gt_filepath)

mesh,vis = init_renderer(ply_file_path)

for key, cam_data in scene_camera_data.items():
    print(key)
    # Estrai i dati della camera
    cam_data = scene_camera_data["" + key]
    cam_R_w2c = cam_data["cam_R_w2c"]
    cam_t_w2c = cam_data["cam_t_w2c"]

    # Estrai i dati del modello
    model_data = scene_gt_data["" + key][0]
    cam_R_m2c = model_data["cam_R_m2c"]
    cam_t_m2c = model_data["cam_t_m2c"]


    cam_matrix_world = set_camera_pose(vis, cam_R_w2c, cam_t_w2c)
    set_model_pose(mesh, cam_R_m2c, cam_t_m2c, cam_matrix_world)
    
    #render
    scene_id = '{:06d}'.format(int(key))
    output_path = basepath + f"\\{scene_id}_000000.png"
    render_ply_to_image(vis, output_path)

# Distruggi la finestra di visualizzazione
vis.destroy_window()

