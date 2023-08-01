from pandaset.dataset import DataSet
import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv
import json
import math
from os.path import *
import os
from tqdm import tqdm

def compose(T, R, Z):
    n = len(T)
    R = np.asarray(R)
    if R.shape != (n,n):
        raise ValueError('Expecting shape (%d,%d) for rotations' % (n,n))
    A = np.eye(n+1)
    ZS = np.diag(Z)
    A[:n,:n] = np.dot(R, ZS)
    A[:n,n] = T[:]
    return A

def quat2mat(q):
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

def _heading_position_to_mat(heading, position):
    quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
    pos = np.array([position["x"], position["y"], position["z"]])
    transform_matrix = compose(np.array(pos), quat2mat(quat), [1.0, 1.0, 1.0])
    return transform_matrix

def lidar_points_to_ego(points, lidar_pose):
    lidar_pose_mat = _heading_position_to_mat(
        lidar_pose['heading'], lidar_pose['position'])
    transform_matrix = np.linalg.inv(lidar_pose_mat)
    return (transform_matrix[:3, :3] @ points.T +  transform_matrix[:3, [3]]).T

def save_pcd(points, save_pcd_file):
    # save_pcd_file = r"D:\Desktop\Project_file\xqm\pandaset\test\lidar_point_cloud_0\30.pcd"
    with open(save_pcd_file, 'w', encoding='ascii') as pcd_file:
        point_num = points.shape[0]
        heads = [
            '# .PCD v0.7 - Point Cloud Data file format',
            'VERSION 0.7',
            'FIELDS x y z intensity',
            'SIZE 4 4 4 1',
            'TYPE F F F U',
            'COUNT 1 1 1 1',
            f'WIDTH {point_num}',
            'HEIGHT 1',
            'VIEWPOINT 0 0 0 1 0 0 0',
            f'POINTS {point_num}',
            'DATA ascii'
        ]
        pcd_file.write('\n'.join(heads))
        for i in range(point_num):
            string_point = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2]) + ' ' + str(
                int(points[i, 3]))
            pcd_file.write(string_point)

def parse_ext(poses):
    lidar_t = np.array([poses['position']['x'], poses['position']['y'], poses['position']['z']]).reshape(3, 1)
    quat = [poses['heading']['x'], poses['heading']['y'], poses['heading']['z'], poses['heading']['w']]
    lidar_r = R.from_quat(quat).as_matrix()
    lidar_ext = np.hstack((lidar_r, lidar_t))
    lidar_ext = np.vstack((lidar_ext, [0, 0, 0, 1]))
    return lidar_ext

def ensure_dir(input_dir):
    if not exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
    return input_dir


def world_system_precessing(seq, dst_dir):
    """以世界坐标系建立点云坐标""" #"""Create point cloud coordinates in world coordinate system"""
    dir_map = {
        "front_left_camera": 'camera_image_0', "front_camera": 'camera_image_1',
        "front_right_camera": 'camera_image_2', "right_camera": 'camera_image_3',
        "back_camera": 'camera_image_4', "left_camera": 'camera_image_5'
    }
    # dst_dir = r"D:\Desktop\Project_file\xqm\pandaset\world_coordinate_system"
    dirs = ['lidar_point_cloud_0', 'camera_config', 'camera_image_0', 'camera_image_1',
            'camera_image_2', 'camera_image_3', 'camera_image_4', 'camera_image_5']
    for _dir in dirs:
        ensure_dir(join(dst_dir, _dir))
    name_num = 0
    i_len = 0
    for p in seq.lidar:
        i_len += 1
    for i in tqdm(range(i_len), desc=f"{seq_name}"):
        name_num += 1
        points = np.array(seq.lidar[i])
        pcd_file = join(dst_dir, 'lidar_point_cloud_0', f"{name_num:0>2}.pcd")
        save_pcd(points, pcd_file)

        lidar_pose = seq.lidar.poses[i]
        lidar_ext = parse_ext(lidar_pose)
        cam_config = []
        for k, v in dir_map.items():
            img = seq.camera[k].data[i]
            img.save(join(dst_dir, v, f"{name_num:0>2}.jpg"))

            cam_pose = seq.camera[k].poses[i]
            cam_ext = parse_ext(cam_pose)
            cam_intrinsics = seq.camera[k].intrinsics
            cam_in = {
                "fx": cam_intrinsics.fx,
                "fy": cam_intrinsics.fy,
                "cx": cam_intrinsics.cx,
                "cy": cam_intrinsics.cy
            }
            cfg_data = {
                "camera_internal": cam_in,
                "camera_external": inv(cam_ext).flatten().tolist()
            }
            cam_config.append(cfg_data)
        cfg_file = join(dst_dir, 'camera_config', f"{name_num:0>2}.json")
        with open(cfg_file, 'w', encoding='utf-8') as f:
            json.dump(cam_config, f)
        i += 1


def car_system_precessing(seq, dst_dir):
    dir_map = {
        "front_left_camera": 'camera_image_0', "front_camera": 'camera_image_1',
        "front_right_camera": 'camera_image_2', "right_camera": 'camera_image_3',
        "back_camera": 'camera_image_4', "left_camera": 'camera_image_5'
    }
    # dst_dir = r"D:\Desktop\Project_file\xqm\pandaset\car_coordinate_system"
    dirs = ['lidar_point_cloud_0', 'camera_config', 'camera_image_0', 'camera_image_1',
            'camera_image_2', 'camera_image_3', 'camera_image_4', 'camera_image_5']
    for _dir in dirs:
        ensure_dir(join(dst_dir, _dir))
    name_num = 0
    i_len = 0
    for p in seq.lidar:
        i_len += 1
    for i in tqdm(range(i_len), desc=f"{seq_name}"):
        name_num += 1
        points = np.array(seq.lidar[i])
        intensity = points[:, 3]
        ego = lidar_points_to_ego(points[:, :3], seq.lidar.poses[i])
        points = np.hstack((ego, intensity.reshape(-1, 1)))
        pcd_file = join(dst_dir, 'lidar_point_cloud_0', f"{name_num:0>2}.pcd")
        save_pcd(points, pcd_file)

        lidar_pose = seq.lidar.poses[i]
        lidar_ext = parse_ext(lidar_pose)
        cam_config = []
        for k, v in dir_map.items():
            img = seq.camera[k].data[i]
            img.save(join(dst_dir, v, f"{name_num:0>2}.jpg"))

            cam_pose = seq.camera[k].poses[i]
            cam_ext = parse_ext(cam_pose)

            config_ext = inv(lidar_ext) @ cam_ext
            cam_intrinsics = seq.camera[k].intrinsics
            cam_in = {
                "fx": cam_intrinsics.fx,
                "fy": cam_intrinsics.fy,
                "cx": cam_intrinsics.cx,
                "cy": cam_intrinsics.cy
            }
            cfg_data = {
                "camera_internal": cam_in,
                "camera_external": inv(config_ext).flatten().tolist()
            }
            cam_config.append(cfg_data)
        cfg_file = join(dst_dir, 'camera_config', f"{name_num:0>2}.json")
        with open(cfg_file, 'w', encoding='utf-8') as f:
            json.dump(cam_config, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pandaset_dir', type=str)
    parser.add_argument('save_dir', type=str)
    parser.add_argument('--is_car_system', default='false', type=str, choices=['true', 'false'],
                        help='Whether to establish a point cloud coordinate system with the acquisition vehicle as the origin')
    args = parser.parse_args()

    pandaset_dir = args.pandaset_dir
    save_dir = args.save_dir
    is_car_system = args.is_car_system
    dataset = DataSet(pandaset_dir)
    for seq_name in dataset.sequences():
        dst_dir = join(save_dir, seq_name)
        seq = dataset[seq_name]
        seq.load()
        if is_car_system == 'true':
            car_system_precessing(seq, dst_dir)
        else:
            world_system_precessing(seq, dst_dir)