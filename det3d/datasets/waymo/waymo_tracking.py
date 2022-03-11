import os.path as osp
import numpy as np
import pickle
import random

from pathlib import Path
from functools import reduce
from typing import Tuple, List
import os 
import json 
from tqdm import tqdm
import argparse

from tqdm import tqdm
try:
    import tensorflow as tf
    tf.enable_eager_execution()
except:
    print("No Tensorflow")

from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


CAT_NAME_TO_ID = {
    'VEHICLE': 1,
    'PEDESTRIAN': 2,
    'SIGN': 3,
    'CYCLIST': 4,
}

CAT_ID_TO_NAME = {
    1: 'VEHICLE',
    2: 'PEDESTRIAN',
    3: 'SIGN',
    4: 'CYCLIST',
}

TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

# ignore sign class 
LABEL_TO_TYPE = {0: 1, 1:2, 2:4}

import uuid 

class UUIDGeneration():
    def __init__(self):
        self.mapping = {}
    def get_uuid(self,seed):
        if seed not in self.mapping:
            self.mapping[seed] = uuid.uuid4().hex 
        return self.mapping[seed]
uuid_gen = UUIDGeneration()

def _create_pd_detection(detections, infos, result_path, tracking=False):
    """Creates a prediction objects file."""
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2

    objects = metrics_pb2.Objects()

    for token, detection in tqdm(detections.items()):
        info = infos[token]
        obj = get_obj(info['anno_path'])

        box3d = detection["box3d_lidar"].detach().cpu().numpy()
        scores = detection["scores"].detach().cpu().numpy()
        labels = detection["label_preds"].detach().cpu().numpy()

        # transform back to Waymo coordinate
        # x,y,z,w,l,h,r2
        # x,y,z,l,w,h,r1
        # r2 = -pi/2 - r1  
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        box3d = box3d[:, [0, 1, 2, 4, 3, 5, -1]]

        if tracking:
            tracking_ids = detection['tracking_ids']

        for i in range(box3d.shape[0]):
            det  = box3d[i]
            score = scores[i]

            label = labels[i]

            o = metrics_pb2.Object()
            o.context_name = obj['scene_name']
            o.frame_timestamp_micros = int(obj['frame_name'].split("_")[-1])

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = det[0]
            box.center_y = det[1]
            box.center_z = det[2]
            box.length = det[3]
            box.width = det[4]
            box.height = det[5]
            box.heading = det[-1]
            o.object.box.CopyFrom(box)
            o.score = score
            # Use correct type.
            o.object.type = LABEL_TO_TYPE[label] 

            if tracking:
                o.object.id = uuid_gen.get_uuid(int(tracking_ids[i]))

            objects.objects.append(o)

    # Write objects to a file.
    if tracking:
        path = os.path.join(result_path, 'tracking_pred.bin')
    else:
        path = os.path.join(result_path, 'detection_pred.bin')

    print("results saved to {}".format(path))
    f = open(path, 'wb')
    f.write(objects.SerializeToString())
    f.close()

def _create_gt_detection(infos, tracking=True):
    """Creates a gt prediction object file for local evaluation."""
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
    
    objects = metrics_pb2.Objects()

    for idx in tqdm(range(len(infos))): 
        info = infos[idx]

        obj = get_obj(info['path'])
        annos = obj['objects']
        num_points_in_gt = np.array([ann['num_points'] for ann in annos])
        box3d = np.array([ann['box'] for ann in annos])

        if len(box3d) == 0:
            continue 

        names = np.array([TYPE_LIST[ann['label']] for ann in annos])

        box3d = box3d[:, [0, 1, 2, 3, 4, 5, -1]]

        for i in range(box3d.shape[0]):
            if num_points_in_gt[i] == 0:
                continue 
            if names[i] == 'UNKNOWN':
                continue 

            det  = box3d[i]
            score = 1.0
            label = names[i]

            o = metrics_pb2.Object()
            o.context_name = obj['scene_name']
            o.frame_timestamp_micros = int(obj['frame_name'].split("_")[-1])

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = det[0]
            box.center_y = det[1]
            box.center_z = det[2]
            box.length = det[3]
            box.width = det[4]
            box.height = det[5]
            box.heading = det[-1]
            o.object.box.CopyFrom(box)
            o.score = score
            # Use correct type.
            o.object.type = CAT_NAME_TO_ID[label]
            o.object.num_lidar_points_in_box = num_points_in_gt[i]
            o.object.id = annos[i]['name']

            objects.objects.append(o)
        
    # Write objects to a file.
    f = open(os.path.join(args.result_path, 'gt_preds.bin'), 'wb')
    f.write(objects.SerializeToString())
    f.close()

def veh_pos_to_transform(veh_pos):
    "convert vehicle pose to two transformation matrix"
    rotation = veh_pos[:3, :3] 
    tran = veh_pos[:3, 3]

    global_from_car = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=False
    )

    car_from_global = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=True
    )

    return global_from_car, car_from_global

###############################################################################################################
############################################## Add by zhanghao ################################################
def transform_box(box, pose):
    """Transforms 3d upright boxes from one frame to another.
    Args:
    box: [..., N, 7] boxes.
    from_frame_pose: [...,4, 4] origin frame poses.
    to_frame_pose: [...,4, 4] target frame poses.
    Returns:
    Transformed boxes of shape [..., N, 7] with the same type as box.
    """
    transform = pose 
    heading = box[..., -1] + np.arctan2(transform[..., 1, 0], transform[..., 0, 0])
    center = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    box[..., 0:3]) + np.expand_dims(
                        transform[..., 0:3, 3], axis=-2)

    velocity = box[..., [6, 7]] 

    velocity = np.concatenate([velocity, np.zeros((velocity.shape[0], 1))], axis=-1) # add z velocity

    velocity = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    velocity)[..., [0, 1]] # remove z axis 

    return np.concatenate([center, box[..., 3:6], velocity, heading[..., np.newaxis]], axis=-1)


def transform_localbox1_to_2(local_box1, veh_to_global1, veh_to_global2):
    """Transform 3d upright boxes from one frame to another

    Args:
        local_box1 (np.array): [..., N, 7], local boxes in frame 1
        veh_to_global1 (np.array): [...,4, 4], pose matrix, frame 1 to global
        veh_to_global2 (np.array): [...,4, 4], pose matrix, frame 2 to global

    Returns:
        Transformed boxes of shape [..., N, 7] with the same type as box.
        Means local box1 projection to the frame2-local-coordinate.
    """
    global_box1 = transform_box(local_box1, veh_to_global1)
    global_to_veh2 = np.linalg.inv(veh_to_global2)
    local_box1_to_2 = transform_box(global_box1, global_to_veh2)
    return local_box1_to_2
###############################################################################################################

def _fill_infos(root_path, frames, split='train', nsweeps=1):
    # load all train infos
    infos = []
    for frame_name in tqdm(frames):  # global id
        lidar_path = os.path.join(root_path, split, 'lidar', frame_name)
        ref_path = os.path.join(root_path, split, 'annos', frame_name)

        ref_obj = get_obj(ref_path)
        ref_time = 1e-6 * int(ref_obj['frame_name'].split("_")[-1])

        ref_pose = np.reshape(ref_obj['veh_to_global'], [4, 4])
        global_from_ref, ref_from_global = veh_pos_to_transform(ref_pose)

        info = {
            "path": lidar_path,
            "anno_path": ref_path, 
            "token": frame_name,
            "timestamp": ref_time,
            "sweeps": []
        }

        sequence_id = int(frame_name.split("_")[1])
        frame_id = int(frame_name.split("_")[3][:-4]) # remove .pkl

        prev_id = frame_id
        sweeps = [] 
        while len(sweeps) < nsweeps - 1:
            if prev_id <= 0:
                if len(sweeps) == 0:
                    sweep = {
                        "path": lidar_path,
                        "token": frame_name,
                        "transform_matrix": None,
                        "time_lag": 0
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                prev_id = prev_id - 1
                # global identifier  

                curr_name = 'seq_{}_frame_{}.pkl'.format(sequence_id, prev_id)
                curr_lidar_path = os.path.join(root_path, split, 'lidar', curr_name)
                curr_label_path = os.path.join(root_path, split, 'annos', curr_name)
                
                curr_obj = get_obj(curr_label_path)
                curr_pose = np.reshape(curr_obj['veh_to_global'], [4, 4])
                global_from_car, _ = veh_pos_to_transform(curr_pose) 
                
                tm = reduce(
                    np.dot,
                    [ref_from_global, global_from_car],
                )

                curr_time = int(curr_obj['frame_name'].split("_")[-1])
                time_lag = ref_time - 1e-6 * curr_time

                sweep = {
                    "path": curr_lidar_path,
                    "transform_matrix": tm,
                    "time_lag": time_lag,
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps

        ##### zhanghao 原本的生成方式
        # if split != 'test':
        #     # read boxes 
        #     TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
        #     annos = ref_obj['objects']
        #     num_points_in_gt = np.array([ann['num_points'] for ann in annos])
        #     gt_boxes = np.array([ann['box'] for ann in annos]).reshape(-1, 9)
            
        #     if len(gt_boxes) != 0:
        #         # transform from Waymo to KITTI coordinate 
        #         # Waymo: x, y, z, length, width, height, rotation from positive x axis clockwisely
        #         # KITTI: x, y, z, width, length, height, rotation from negative y axis counterclockwisely 
        #         gt_boxes[:, -1] = -np.pi / 2 - gt_boxes[:, -1]
        #         gt_boxes[:, [3, 4]] = gt_boxes[:, [4, 3]]

        #     gt_names = np.array([TYPE_LIST[ann['label']] for ann in annos])
        #     mask_not_zero = (num_points_in_gt > 0).reshape(-1)    

        #     # filter boxes without lidar points 
        #     info['gt_boxes'] = gt_boxes[mask_not_zero, :].astype(np.float32)
        #     info['gt_names'] = gt_names[mask_not_zero].astype(str)

        tokens_list = []
        gt_boxes_list = []
        prev_boxes_list = []
        names_list = []
        prev_names_list = []
        
        curr_annos = ref_obj['objects']
        curr_pose = np.reshape(ref_obj['veh_to_global'], [4, 4])
        curr_obj_dict = {}
        for obj in curr_annos:
            curr_obj_dict.update({
                obj['name'] : obj
            })

        frame_id = int(frame_name.split("_")[3][:-4]) # remove .pkl
        prev_obj_dict = {}
        prev_pose = np.eye(4)
        if frame_id > 0:
            # 有前帧数据
            prev_id = frame_id - 1
            prev_name = 'seq_{}_frame_{}.pkl'.format(sequence_id, prev_id)
            prev_label_path = os.path.join(root_path, split, 'annos', prev_name)
            prev_obj = get_obj(prev_label_path)
            prev_annos = prev_obj['objects']
            prev_pose = np.reshape(prev_obj['veh_to_global'], [4, 4])  
            for obj in prev_annos:
                prev_obj_dict.update({
                    obj['name'] : obj
                })

        for prev_obj_id_name, prev_obj in prev_obj_dict.items():
            # 前帧的annos
            prev_obj_local_box = np.expand_dims(prev_obj['box'], 0)
            prev_obj_box_in_curr = transform_localbox1_to_2(prev_obj_local_box, prev_pose, curr_pose)[0]

            prev_boxes_list.append(prev_obj_box_in_curr[[0,1,2,3,4,5,8]])  # without speed
            prev_names_list.append(CAT_ID_TO_NAME[prev_obj['label']])

            if prev_obj_id_name in curr_obj_dict:
                # 前帧的目标在本帧仍然存在
                gt_boxes_list.append(curr_obj_dict[prev_obj_id_name]['box'])
                names_list.append(CAT_ID_TO_NAME[curr_obj_dict[prev_obj_id_name]['label']])
                tokens_list.append(prev_obj_id_name)
            else:
                # 前帧的目标在本帧消失了
                gt_boxes_list.append(np.array([np.nan]*9))
                names_list.append('disappear')
                tokens_list.append('')

        for curr_obj_id_name, curr_obj in curr_obj_dict.items():
            if curr_obj_id_name in tokens_list:
                continue
            else:
                prev_boxes_list.append(np.array([np.nan]*7))
                prev_names_list.append('new_obj')

                gt_boxes_list.append(curr_obj_dict[curr_obj_id_name]['box'])
                names_list.append(CAT_ID_TO_NAME[curr_obj_dict[curr_obj_id_name]['label']])
                tokens_list.append(curr_obj_id_name)

        if len(gt_boxes_list) > 0:
            info["gt_boxes"] = np.stack(gt_boxes_list)
            info["gt_names"] = np.array(names_list)
            info["gt_boxes_token"] = tokens_list
            info['prev_gt_boxes'] = np.stack(prev_boxes_list)
            info['prev_gt_names'] = np.array(prev_names_list)
        else:
            info["gt_boxes"] = np.empty((0,9))
            info["gt_names"] = np.array([])
            info["gt_boxes_token"] = []
            info['prev_gt_boxes'] = np.empty((0,7))
            info['prev_gt_names'] = np.array([])

        infos.append(info)
    return infos

def sort_frame(frames):
    indices = [] 

    for f in frames:
        seq_id = int(f.split("_")[1])
        frame_id= int(f.split("_")[3][:-4])

        idx = seq_id * 1000 + frame_id
        indices.append(idx)

    rank = list(np.argsort(np.array(indices)))

    frames = [frames[r] for r in rank]
    return frames

def get_available_frames(root, split):
    dir_path = os.path.join(root, split, 'lidar')
    available_frames = list(os.listdir(dir_path))

    sorted_frames = sort_frame(available_frames)

    print(split, " split ", "exist frame num:", len(available_frames))
    return sorted_frames


def create_waymo_tracking_infos(root_path, split='train', nsweeps=1):
    frames = get_available_frames(root_path, split)

    waymo_infos = _fill_infos(
        root_path, frames, split, nsweeps
    )

    print(
        f"sample: {len(waymo_infos)}"
    )
    with open(
        os.path.join(root_path, "simtrack_infos_"+split+"_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps)), "wb"
    ) as f:
        pickle.dump(waymo_infos, f)

def parse_args():
    parser = argparse.ArgumentParser(description="Waymo 3D Extractor")
    parser.add_argument("--path", type=str, default="data/Waymo/tfrecord_training")
    parser.add_argument("--info_path", type=str)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--gt", action='store_true' )
    parser.add_argument("--tracking", action='store_true')
    args = parser.parse_args()
    return args


def reorganize_info(infos):
    new_info = {}

    for info in infos:
        token = info['token']
        new_info[token] = info

    return new_info 

if __name__ == "__main__":
    args = parse_args()

    with open(args.info_path, 'rb') as f:
        infos = pickle.load(f)
    
    if args.gt:
        _create_gt_detection(infos, tracking=args.tracking)
        exit() 

    infos = reorganize_info(infos)
    with open(args.path, 'rb') as f:
        preds = pickle.load(f)
    _create_pd_detection(preds, infos, args.result_path, tracking=args.tracking)
