"""
python tools/val_waymo_tracking.py \
    --config experiments/pointpillars/configs/waymo_pp_centernet_tracking.py \
    --work_dir work_dir/waymo/track_results \
    --checkpoint work_dir/waymo/epoch_32.pth \
    --eval_det
"""
import os
import cv2
import time
import copy
import pickle 
import argparse
import numpy as np

import torch
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config

from det3d.torchie.trainer import load_checkpoint
from det3d.torchie.trainer.trainer import example_to_device
from det3d.torchie.trainer.utils import all_gather, synchronize
from det3d.core.utils.center_utils import (draw_gaussian, gaussian_radius)
from det3d.core.bbox.box_np_ops import center_to_corner_box2d
from det3d.core.bbox.geometry import points_in_convex_polygon_jit

# from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from eval_track import calcMetrics


# TEST_WAYMO_SEQS = range(1)
TEST_WAYMO_SEQS = range(202)


def parse_args():
    parser = argparse.ArgumentParser(description="Waymo Tracking")
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save results")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--eval_det", action='store_true')

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def get_pose_from_token(token, anno_path="/mnt/data/waymo_opensets/val/annos"):
    data = pickle.load(open(os.path.join(anno_path, token), "rb"))
    return data['veh_to_global'].reshape(4,4)


def ref_vel_to_global(ref_vel, global_from_ref_rotation):
    ref_vel = np.concatenate((ref_vel, np.zeros((ref_vel.shape[0], 1)) ), axis=1)
    global_vel = np.matmul(ref_vel, Quaternion(matrix=global_from_ref_rotation).rotation_matrix.T)[:,:2]
    return global_vel


def save_track_res_txt(save_dir, output, global_from_ref_rotation):
    # veh_to_global = get_pose_from_token(output['metadata']['token'])
    # global_from_ref_rotation = veh_to_global[:3, :3]

    bbb = output['box3d_lidar'].clone().detach().cpu().numpy()
    bbb[:, -3:-1] = ref_vel_to_global(bbb[:, -3:-1], global_from_ref_rotation)
    sss = output['scores'].clone().detach().cpu().numpy().reshape(-1,1)
    lll = output['label_preds'].clone().detach().cpu().numpy().reshape(-1,1)
    iii = output['tracking_id'].clone().detach().cpu().numpy().reshape(-1,1)
    all_data_npy = np.concatenate([iii, lll, bbb, sss], axis=1)
    all_data_npy = all_data_npy[:, [0,1, 2,3,4,5,6,7,10, 11, 8,9]]

    # np.savetxt(save_dir + "/" + output['metadata']['token'].replace(".pkl", ".txt"), all_data_npy.sort(axis=0))
    np.savetxt(save_dir + "/" + output['metadata']['token'].replace(".pkl", ".txt"), all_data_npy)


def vis_heatmap(hm, save_path = "hm.jpg"):
    hm_np = np.array(copy.deepcopy(hm)).transpose(1,2,0)
    hm_np *= 255.0
    # print("saving to : ", save_path, hm_np.shape)
    cv2.imwrite(save_path + ".jpg", np.array(hm_np))


def tracking():
    args = parse_args()

    txt_save_dir = os.path.join(args.work_dir, "txt")
    if not os.path.exists(txt_save_dir):
        os.makedirs(txt_save_dir, exist_ok=True)
    os.system("cp %s %s"%(args.config, args.work_dir))

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # cfg._voxel_size = (0.32, 0.32, 6.0)
    # cfg._pc_range = (-74.88, -74.88, -2, 74.88, 74.88, 4.0)
    
    global voxel_size, downsample, voxel_range, num_classes, size_h, size_w
    voxel_size = np.array(cfg._voxel_size)[:2]
    downsample= cfg.assigner.out_size_factor
    voxel_range = np.array(cfg._pc_range)
    num_classes = sum([t['num_class'] for t in cfg.tasks])
    # size_w, size_h = ((voxel_range[3:5] - voxel_range[:2]) / voxel_size  / downsample).astype(np.int32)
    size_w, size_h = 468, 468
    print("size_w, size_h = ", size_w, size_h)
        
    dataset = build_dataset(cfg.data.val)
    
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, args.checkpoint, map_location="cpu")
    print("Done loading checkpoints: ", args.checkpoint)
    
    data_loader = build_dataloader(dataset, batch_size=1, workers_per_gpu=8, dist=False, shuffle=False,)

    model = model.cuda()
    model.eval()

    cpu_device = torch.device("cpu")

    prev_detections = {}
    grids = meshgrid(size_w, size_h)

    print("Data_loader.size = ", len(data_loader))
    start_id = 0
    with torch.no_grad():
        for iidx, data_batch in enumerate(data_loader):
            device = torch.device(args.local_rank)
            data_batch = example_to_device(data_batch, device, non_blocking=False)

            # print(data_batch['metadata'])
            curr_token = data_batch['metadata'][0]['token']
            curr_frame_id = int(curr_token.split("_")[-1][:-4])
            seq_id = curr_token.split("_")[1]

            if int(seq_id) not in TEST_WAYMO_SEQS:
                print(seq_id, "out of TEST_WAYMO_SEQS, finished.")
                break

            # t1 = time.time()
            print("start ", curr_token)

            track_outputs = None
            # if  prev_token != '': # non-first frame
            if curr_frame_id > 0: # non-first frame
                prev_token = "seq_" + seq_id + "_frame_" + str(curr_frame_id - 1) + ".pkl"

                assert prev_token in prev_detections.keys()
                box3d = prev_detections[prev_token]['box3d_global']
                # box3d = (data_batch['ref_from_car'][0].detach().numpy() @ data_batch['car_from_global'][0].detach().numpy()) @ box3d
                curr_veh_to_global = get_pose_from_token(curr_token)
                box3d = np.linalg.inv(curr_veh_to_global) @ box3d
                box3d = box3d.T
                prev_detections[prev_token]['box3d_lidar'] = np.concatenate((box3d[:, :3], 
                                                            prev_detections[prev_token]['box3d_lidar'][:, 3:]), axis=1)

                prev_hm_, prev_track_id_ = render_trackmap(prev_detections[prev_token],  grids, cfg)
                prev_hm_ = prev_hm_.permute(0,2,3,1).view(1, size_h*size_w, num_classes).contiguous().to(device, non_blocking=False)
                prev_track_id_ = prev_track_id_.permute(0,2,3,1).view(1, size_h*size_w, num_classes).contiguous().to(device, non_blocking=False)
                
                # t2 = time.time()
                # print("done render prev_hm: %.2f s"%(t2 - t1))

                prev_hm = []
                prev_track_id = []
                class_id = 0
                for task in cfg.tasks:
                    prev_hm.append(prev_hm_[..., class_id : class_id+task['num_class']])
                    prev_track_id.append(prev_track_id_[..., class_id : class_id+task['num_class']])
                    class_id += task['num_class']

                preds = model(data_batch, return_loss=False, return_feature=True)

                # t3 = time.time()
                # print("done model forward: %.2f s"%(t3 - t2))

                outputs, track_outputs = model.bbox_head.predict_tracking(data_batch, preds, model.test_cfg, prev_hm = prev_hm, prev_track_id=prev_track_id, new_only=False)
                outputs[0]['tracking_id'] = torch.arange(start_id, start_id + outputs[0]['scores'].size(0)).int()
                start_id += outputs[0]['scores'].size(0) 

                # t4 = time.time()
                # print("done model predict_tracking: %.2f s"%(t4 - t3))

            else: 
                # first frame
                outputs = model(data_batch, return_loss=False)
                outputs[0]['tracking_id'] = torch.arange(start_id, start_id + outputs[0]['scores'].size(0)).int()
                start_id += outputs[0]['scores'].size(0)
            
            output = copy.deepcopy(outputs[0])
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in ["metadata"]:
                    if track_outputs is not None:
                        output[k] = torch.cat([v.clone().to(cpu_device), track_outputs[0][k].clone().to(cpu_device)], dim=0)
                    else:
                        output[k] = v.clone().to(cpu_device)

            # save output to txt
            veh_to_global = get_pose_from_token(output['metadata']['token'])
            save_track_res_txt(txt_save_dir, output, veh_to_global[:3, :3])
            
            prev_output = {}
            box3d_lidar = output['box3d_lidar'].clone().detach().cpu().numpy()
            box3d = np.concatenate((box3d_lidar[:, :3], np.ones((box3d_lidar.shape[0],1))), axis=1).T
            box3d = veh_to_global @ box3d
            prev_output['box3d_lidar'] = box3d_lidar
            prev_output['box3d_global'] = box3d
            prev_output['label_preds'] = output['label_preds'].cpu().numpy()
            prev_output['scores'] = output['scores'].cpu().numpy()
            prev_output['tracking_id'] = output['tracking_id'].cpu().numpy()
            prev_detections[output['metadata']['token']] = prev_output
    
    # eval tracking
    if args.eval_det:
        calcMetrics(tk_path=txt_save_dir, seqs=TEST_WAYMO_SEQS)


def render_trackmap(preds_dicts, grids, cfg):
    prev_hm = np.zeros((1, num_classes, size_h, size_w),dtype=np.float32)
    prev_tracking_map = np.zeros((1, num_classes, size_h, size_w), dtype=np.int64) - 1
    label_preds = preds_dicts['label_preds']
    box3d_lidar = preds_dicts['box3d_lidar']
    scores = preds_dicts['scores']
    tracking_ids = preds_dicts['tracking_id']
    
    box_corners = center_to_corner_box2d(box3d_lidar[:, :2], box3d_lidar[:, 3:5], box3d_lidar[:, -1])
    box_corners = (box_corners - voxel_range[:2].reshape(1, 1, 2)) / voxel_size[:2].reshape(1, 1, 2) / downsample
    masks = points_in_convex_polygon_jit(grids, box_corners)
    
    for obj in range(label_preds.shape[0]):
        cls_id = label_preds[obj]
        score = scores[obj]
        tracking_id = tracking_ids[obj]
        size_x, size_y = box3d_lidar[obj, 3] / voxel_size[0] / downsample, box3d_lidar[obj, 4] / voxel_size[1] / downsample
        if size_x > 0 and size_y > 0:
            radius = gaussian_radius((size_y, size_x), min_overlap=0.1)
            radius = min(cfg.assigner.min_radius, int(radius))

            coor_x = (box3d_lidar[obj, 0] - voxel_range[0]) / voxel_size[0] / downsample
            coor_y = (box3d_lidar[obj, 1] - voxel_range[1]) / voxel_size[1] / downsample
            ct = np.array([coor_x, coor_y], dtype=np.float32)  
            ct_int = ct.astype(np.int32)
            # throw out not in range objects to avoid out of array area when creating the heatmap
            if not (0 <= ct_int[0] < size_w and 0 <= ct_int[1] < size_h):
                continue 
            # render center map as in centertrack
            draw_gaussian(prev_hm[0, cls_id], ct, radius, score)  #

            # tracking ID map
            mask = masks[:, obj].nonzero()[0]
            coord_in_box = grids[mask, :]
            mask1 = prev_tracking_map[0, cls_id][coord_in_box[:, 1], coord_in_box[:, 0]] == -1
            mask2 = prev_hm[0, cls_id][coord_in_box[:, 1], coord_in_box[:, 0]] < score
            mask = mask[np.logical_or(mask1, mask2)]
            coord_in_box = grids[mask, :]
            prev_tracking_map[0, cls_id][coord_in_box[:, 1], coord_in_box[:, 0]] = tracking_id
            prev_tracking_map[0, cls_id][ct_int[1], ct_int[0]] = tracking_id
           
    return torch.from_numpy(prev_hm), torch.from_numpy(prev_tracking_map)
   
def meshgrid(w, h):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = ww.reshape(-1) 
    hh = hh.reshape(-1)

    return np.stack([ww, hh], axis=1)


if __name__ == "__main__":
    tracking()