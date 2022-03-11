import numpy as np
from pathlib import Path

from det3d.core import box_np_ops
from ..registry import PIPELINES
import pickle
import os

#####################################
############ WaymoDataset ###########
def get_obj(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj 

def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    return points

def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T
#####################################


#####################################
########## NuscenesDataset ##########
def read_file(path, num_point_feature=4):
    points = np.fromfile(path, dtype=np.float32)
    points = points.reshape(-1, 5)[:, :num_point_feature]
    return points

def read_sweep(sweep):
    min_distance = 1.0
    # points_sweep = np.fromfile(str(sweep["lidar_path"]),
    #                            dtype=np.float32).reshape([-1,
    #                                                       5])[:, :4].T
    points_sweep = read_file(str(sweep["lidar_path"])).T

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    points_sweep = remove_close(points_sweep, min_distance)
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T

def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points
#####################################


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset, **kwargs):
        self.type = dataset
        self.nsweeps = kwargs.get('nsweeps', -1)

    def __call__(self, res, info):

        res["type"] = self.type

        if self.type == "NuScenesDataset":

            lidar_path = info["lidar_path"]
           
            points = read_file(str(lidar_path))

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            for i in range(len(info["sweeps"])):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep)
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = np.hstack([points, times])
           
        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            # res["lidar"]["points"] = points

            if nsweeps > 1: 
                # sweep_points_list = [points]
                # sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = np.hstack([points, times])
            # res["lidar"]["points"] = points
            # res["lidar"]["times"] = times
            # res["lidar"]["combined"] = np.hstack([points, times])

        else:
            raise NotImplementedError
        # print("Loading pointcloud done...")
        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, res, info):
        if res["type"] in ["NuScenesDataset", "LyftDataset"] and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
            }

            if 'prev_gt_boxes' in info:
                res["lidar"]["annotations"].update(dict(
                    prev_gt_boxes=info['prev_gt_boxes'].astype(np.float32),
                    prev_gt_names=info['prev_gt_names'],))

        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
            if 'prev_gt_boxes' in info:
                res["lidar"]["annotations"].update(dict(
                    prev_gt_boxes=info['prev_gt_boxes'].astype(np.float32),
                    prev_gt_names=info['prev_gt_names'],))
        else:
            raise NotImplementedError
        # print("Loading annotations done...")
        return res, info
