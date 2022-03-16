import os
import pickle
from pathlib import Path

import numpy as np
from det3d.core import box_np_ops
from det3d.datasets.dataset_factory import get_dataset
from tqdm import tqdm

dataset_name_map = {
    "NUSC": "NuScenesDataset",
    "LYFT": "LyftDataset",
    "WAYMO": "WaymoDataset",
}


def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_path=None,
    used_classes=None,
    db_path=None,
    dbinfo_path=None,
    relative_path=True,
    **kwargs,
):
    pipeline = [
        {
            "type": "LoadPointCloudFromFile",
            "dataset": dataset_name_map[dataset_class_name],
        },
        {"type": "LoadPointCloudAnnotations", "with_bbox": True},
    ]

    if "nsweeps" in kwargs:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path,
            root_path=data_path,
            pipeline=pipeline,
            test_mode=True,
            nsweeps=kwargs["nsweeps"],
        )
        nsweeps = dataset.nsweeps
    else:
        dataset = get_dataset(dataset_class_name)(
            info_path=info_path, root_path=data_path, test_mode=True, pipeline=pipeline
        )
        nsweeps = 1

    root_path = Path(data_path)

    if dataset_class_name in ["WAYMO", "NUSC"]: 
        if db_path is None:
            db_path = root_path / f"gt_database_{nsweeps}sweeps_withvelo"
        if dbinfo_path is None:
            dbinfo_path = root_path / f"dbinfos_train_{nsweeps}sweeps_withvelo.pkl"
    else:
        raise NotImplementedError()

    if dataset_class_name == "NUSC":
        point_features = 5
    elif dataset_class_name == "WAYMO":
        point_features = 5 if nsweeps == 1 else 6 
    else:
        raise NotImplementedError()

    print("Using point_features : ", point_features)
    print("Using nsweeps : ", dataset.nsweeps)

    db_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    group_counter = 0

    print("len(dataset) = ", len(dataset))
    for index in tqdm(range(len(dataset))):
        image_idx = index
        # modified to nuscenes
        sensor_data = dataset.get_sensor_data(index)
        if "image_idx" in sensor_data["metadata"]:
            image_idx = sensor_data["metadata"]["image_idx"]
        # if nsweeps > 1: 
        #     points = sensor_data["lidar"]["combined"]
        # else:
        #     points = sensor_data["lidar"]["points"]
        points = sensor_data["lidar"]["points"]

        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]
        names = annos["names"]

        if dataset_class_name == 'WAYMO':
            # waymo dataset contains millions of objects and it is not possible to store
            # all of them into a single folder
            # we randomly sample a few objects for gt augmentation
            # We keep all cyclist as they are rare 
            if index % 4 != 0:
                mask = (names == 'VEHICLE') 
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]

            if index % 2 != 0:
                mask = (names == 'PEDESTRIAN')
                mask = np.logical_not(mask)
                names = names[mask]
                gt_boxes = gt_boxes[mask]


        group_dict = {}
        group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]
        
        
        
        num_obj = gt_boxes.shape[0]
        if num_obj == 0 or len(gt_boxes.shape)!=2:
        # if num_obj == 0:    # zhanghao
            print("token = ", sensor_data["metadata"]["token"])
            print("gt_boxes.shape = ", gt_boxes.shape)
            continue 
            
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
        for i in range(num_obj):
            if (used_classes is None) or names[i] in used_classes:
                filename = f"{image_idx}_{names[i]}_{i}.bin"
                dirpath = os.path.join(str(db_path), names[i])
                os.makedirs(dirpath, exist_ok=True)

                filepath = os.path.join(str(db_path), names[i], filename)

                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "w") as f:
                    try:
                        gt_points[:, :point_features].tofile(f)
                    except:
                        print("process {} files".format(index))
                        break

            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_dump_path = os.path.join(db_path.stem, names[i], filename)
                else:
                    db_dump_path = str(filepath)

                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]


    print("dataset length: ", len(dataset))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, "wb") as f:
        pickle.dump(all_db_infos, f)



# def create_groundtruth_database(
#     dataset_class_name,
#     data_path,
#     info_path,
#     nsweeps,
#     used_classes=None,
#     db_path=None,
#     dbinfo_path=None,
#     relative_path=True,
#     **kwargs,
# ):
   
#     root_path = Path(data_path)

#     if db_path is None:
#         db_path = root_path / f"gt_database_{nsweeps}sweeps"
#     if dbinfo_path is None:
#         dbinfo_path = root_path / f"dbinfos_train_{nsweeps}sweeps.pkl"
  
#     if dataset_class_name == "NUSC" or dataset_class_name == "LYFT" or dataset_class_name == 'WAYMO':
#         point_features = 5

#     db_path.mkdir(parents=True, exist_ok=True)

#     all_db_infos = {}
#     group_counter = 0

#     with open(info_path, "rb") as f:
#         _nusc_infos_all = pickle.load(f)
    
#     for idx in tqdm(range(len(_nusc_infos_all))):
#         info = _nusc_infos_all[idx]
        
#         points = np.fromfile(info["lidar_path"], dtype=np.float32).reshape(-1, 5)

#         gt_boxes = info["gt_boxes"]
#         names = info["gt_names"]

#         group_dict = {}
#         group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        
#         group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
#         difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)

#         num_obj = gt_boxes.shape[0]
#         if num_obj == 0:
#             continue 
#         point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)
#         for i in range(num_obj):
#             if (used_classes is None) or names[i] in used_classes:
#                 filename = f"{names[i]}_{i}.bin"
#                 filepath = db_path / filename
#                 gt_points = points[point_indices[:, i]]
#                 gt_points[:, :3] -= gt_boxes[i, :3]
#                 with open(filepath, "w") as f:
#                     try:
#                         gt_points[:, :point_features].tofile(f)
#                     except:
#                         print("process {} files".format(idx))
#                         break

#             if (used_classes is None) or names[i] in used_classes:
#                 if relative_path:
#                     db_dump_path = str(db_path.stem + "/" + filename)
#                 else:
#                     db_dump_path = str(filepath)

#                 db_info = {
#                     "name": names[i],
#                     "path": db_dump_path,
#                     "image_idx": root_path,
#                     "gt_idx": i,
#                     "box3d_lidar": gt_boxes[i],
#                     "num_points_in_gt": gt_points.shape[0],
#                     "difficulty": difficulty[i],
#                     # "group_id": -1,
#                     # "bbox": bboxes[i],
#                 }
#                 local_group_id = group_ids[i]
#                 # if local_group_id >= 0:
#                 if local_group_id not in group_dict:
#                     group_dict[local_group_id] = group_counter
#                     group_counter += 1
#                 db_info["group_id"] = group_dict[local_group_id]
#                 if names[i] in all_db_infos:
#                     all_db_infos[names[i]].append(db_info)
#                 else:
#                     all_db_infos[names[i]] = [db_info]

#     for k, v in all_db_infos.items():
#         print(f"load {len(v)} {k} database infos")

#     with open(dbinfo_path, "wb") as f:
#         pickle.dump(all_db_infos, f)



if __name__ == "__main__":
    create_groundtruth_database(
        "WAYMO",
        "./data/Waymo",
        Path("./data/Waymo") / "simtrack_infos_train_{:02d}sweeps_filter_zero_gt.pkl".format(2),
        used_classes=['VEHICLE', 'PEDESTRIAN'],
        nsweeps=2
    )
