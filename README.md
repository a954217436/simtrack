# 一、复现

开源代码只支持 nuscenes dataset，为了与已实验的算法进行对比，需要在 waymo dataset 进行复现。

## 1.1 waymo 数据集处理

需要按照 nuscenes 的处理方法，生成 2-sweeps 的 info 以供训练时使用，这里需要注意的是，该 info 与 centerpoint 中的 info 差异较大，需要额外填充如下几个 keys:

- gt_boxes
- gt_names
- prev_gt_boxes
- prev_gt_names

另外，需要对类别进行重新梳理，根据目标的状态，添加 **"disappear"， "new_obj"** 等类别。
核心复现代码如下：

```python
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
```

- 注意：目前的实现，未对 waymo 坐标系到 kitti 坐标系的转换，需要进一步确定这步是否必须要做（det3d 框架是基于 kitti 坐标系开发的）。

## 1.2 waymo 训练配置文件

第一步的转换完成后，得到 simtrack_infos_train_02sweeps_filter_zero_gt.pkl 和 simtrack_infos_val_02sweeps_filter_zero_gt.pkl。需要对 waymo 写单独的配置文件，以便开展训练，配置文件如下：

```python
WAYMO_PATH = "/mnt/data/waymo_tiny_simtrack/"
tasks = [
    dict(num_class=2, class_names=['VEHICLE', 'PEDESTRIAN'], stride=1),
]
_voxel_size = (0.32, 0.32, 6.0)
_pc_range = (-74.88, -74.88, -2, 74.88, 74.88, 4.0)
# model settings
model = dict(
    type="PointPillars",
    pretrained=None,
    reader=dict(
        type="PillarFeatureNet",
        num_input_features=6,
        num_filters=[64, 64],
        with_distance=False,
        voxel_size=_voxel_size,
        pc_range=_pc_range,
        norm_cfg=norm_cfg,
    ),
    backbone=dict(type="PointPillarsScatter", num_input_features=64, norm_cfg=norm_cfg, ds_factor=1),
    neck=dict(
        type="RPN",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[1, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[1, 2, 4], # #[1, 2, 4], #, 0.5, 1, 2
        us_num_filters=[128, 128, 128],
        num_input_features=64,
        norm_cfg=norm_cfg,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHeadV2",
        in_channels=sum([128, 128, 128]),  # this is linked to 'neck' us_num_filters
        tasks=tasks,
        dataset='waymo',
        weight=0.25,
        code_weights=[4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)},
    ),
)

target_assigner = dict(tasks=tasks)
assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


# dataset settings
dataset_type = "WaymoDataset"
n_sweeps = 2
data_root = WAYMO_PATH


train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    global_trans_noise=[0.2, 0.2, 0.2],
    remove_points_after_sample=False,
    remove_unknown_examples=False,
    min_points_in_gt=0,
    flip=[0.5, 0.5],
    db_sampler=db_sampler,
    class_names=class_names,
)


train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, nsweeps=n_sweeps),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTracking", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, nsweeps=n_sweeps),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="Reformat"),
]

train_anno = WAYMO_PATH + "/simtrack_infos_train_02sweeps_filter_zero_gt.pkl"
val_anno   = WAYMO_PATH + "/simtrack_infos_val_02sweeps_filter_zero_gt.pkl"
test_anno = None

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)

"""training hooks """
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy in training hooks
lr_config = dict(
    type="one_cycle", lr_max=0.004, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

use_syncbn = True
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
# yapf:enable
# runtime settings
total_epochs = 36
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
# work_dir = "./experiments/pointpillars"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None
workflow = [("train", 1), ("val", 1)]
```

# 二、实验记录

## 2.1 实验

目前已进行了多组实验，其中结果较好的两组实验如下：
实验一参数配置：

| **参数项**                    | **参数值**                                                                        |
| ----------------------------- | --------------------------------------------------------------------------------- |
| n_sweeps                      | 2                                                                                 |
| classes                       | ['VEHICLE', 'PEDESTRIAN']                                                         |
| voxel_size                    | (0.32, 0.32, 6.0)                                                                 |
| pc_range                      | (-74.88, -74.88, -2, 74.88, 74.88, 4.0)                                           |
| PillarFeatureNet: num_filters | [64, 64]                                                                          |
| RPN                           | [3, 5, 5], [1, 2, 2], [128, 128, 128]                                             |
| CenterHeadV2: code_weights    | [4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0]                                |
| db_gt_sampler                 | False                                                                             |
| total_epochs                  | 32                                                                                |
| lr_config                     | type="one_cycle", lr_max=0.004, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4 |

实验二参数配置：

| **参数项**                    | **参数值**                                                                        |
| ----------------------------- | --------------------------------------------------------------------------------- |
| n_sweeps                      | 2                                                                                 |
| classes(2-heads)              | ['VEHICLE'], ['PEDESTRIAN']                                                       |
| voxel_size                    | (0.2, 0.2, 6.0)                                                                   |
| pc_range                      | (-75.2, -75.2, -2, 75.2, 75.2, 4.0)                                               |
| PillarFeatureNet: num_filters | [64, 64]                                                                          |
| RPN                           | [3, 5, 5], [1, 2, 2], [128, 128, 128]                                             |
| CenterHeadV2: code_weights    | [4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0]                                |
| db_gt_sampler                 | False                                                                             |
| total_epochs                  | 36                                                                                |
| lr_config                     | type="one_cycle", lr_max=0.004, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4 |

### 2.1.1 Detection mAP

实验一的 32 个 epoch 训练完成后，在 waymo 验证集上测试 mAP，结果如下，为了便于分析，表中给出了 centerpoint 的一些指标。

| **模型**                                                                     | **验证参数**                                | **VEHICLE_LEVEL_2/mAP** | **VEHICLE_LEVEL_2/mAPH** | **VEHICLE_LEVEL_2 Recall@0.95** | **PEDESTRIAN_LEVEL_2/mAP** | **PEDESTRIAN_LEVEL_2/mAPH** | **PEDESTRIAN_LEVEL_2** **Recall@0.95** |
| ---------------------------------------------------------------------------- | ------------------------------------------- | ----------------------- | ------------------------ | ------------------------------- | -------------------------- | --------------------------- | -------------------------------------- |
| **实验一** **simtrack_pp_two_cls_epoch32** **(2 sweeps, voxel=0.32)**        | thresh=0.2,box=TYPE_2D,iou_thresh=[0.7,0.5] | 0.75064474              | 0.7406002                | 0.6000373                       | 0.66397935                 | 0.6066588                   | 0.3614851                              |
| **实验二** **simtrack_pp_two_cls_epoch36** **(2 sweeps, voxel=0.2, 2heads)** | thresh=0.2,box=TYPE_2D,iou_thresh=[0.7,0.5] | 0.7087538               | 0.692822                 | 0.59762186                      | 0.68568736                 | 0.6274929                   | 0.43303108                             |
| waymo_centerpoint_pp_two_pfn_stride1_3x_epoch_36 (1 sweep)                   | thresh=0.2,box=TYPE_2D,iou_thresh=[0.7,0.5] | 0.7405185               | 0.7319974                | 0.5260502                       | 0.6105477                  | 0.50858194                  | 0.21582854                             |
| waymo_centerpoint_voxelnet_3x_epoch_36(1 sweep)                              | thresh=0.2,box=TYPE_2D,iou_thresh=[0.7,0.5] | 0.7535514               | 0.7450954                | 0.5848034                       | 0.66580003                 | 0.6019306                   | 0.183324                               |
| waymo_centerpoint_voxelnet_2sweep_3x_epoch_36(2 sweeps)                      | thresh=0.2,box=TYPE_2D,iou_thresh=[0.7,0.5] | 0.76343805              | 0.7555981                | 0.54180664                      | 0.6910058                  | 0.65268767                  | 0.17844498                             |

### 2.1.2 Tracking MOTA

使用 epoch 32，在 waymo 验证集上测试 MOTA 指标，结果如下，为了便于分析，表中给出了之前所做的基于 KF 与 centerpoint 的一些指标。
在 waymo validation 测试集，共 202 sequences，39987 frames，16912 条轨迹，1787380 个目标。


| **Metric** | **KF-CA 优化 1** | **Centerpoint tracking** | **Simtrack 实验一** **(pp_2cls_epoch32, 2sweeps,voxel_size=(0.32,0.32),nms_cfg=[2048,500,0.1])**| **Simtrack 实验二pp_2cls_epoch36, 2sweeps,voxel_size=(0.2,0.2),****nms_cfg=[1024,256,0.2])** |
| --- | --- | --- | --- | --- |
| MOTA（总指标） | 0.6637 | 0.6524 | 0.6868 | 0.6770 |
| TP（True Positive） | 1346584 | 1291394 | 1314345 | 1358232 |
| FP（False Positive，即误报） | 157908 | 121124 | 86602 | 148044 |
| FN（False Negative，即漏报） | 440796 | 495986 | 473035 | 429148 |
| Recall（召回率） | 0.7534 | 0.7225 | 0.7353 | 0.7599 |
| Precision（精度） | 0.8950 | 0.9142 | 0.9382 | 0.9017 |
| ID-Switch | 2460 | 4172 | 154 | 171 |
| Tracker Trajectories（跟踪轨迹数量） | 28423 | 20405 | 32245 | 47896 |
| AOE（角度误差, 弧度） | 0.1497 | 0.1187 | 0.1614 | 0.1896 |
| ASE（尺度误差, m） | 0.1679 | 0.1685 | 0.1779 | 0.1824 |
| ATE（偏移误差, m） | 0.1510 | 0.1396 | 0.1485 | 0.1570 |
| AVE（速度误差, km/h） | 0.6212 | 0.4702 | 0.5531 | 0.5913 |
| AVE(0-10, km/h) | 0.4688 | 0.2963 | 0.3482 | 0.3719 |
| AVE(10-40, km/h) | 1.6616 | 1.6338 | 1.9142 | 2.0453 |
| AVE(40-60, km/h) | 2.0377 | 2.0074 | 2.2636 | 2.5726 |
| AVE(60-80, km/h) | 2.3755 | 1.9279 | 2.4299 | 2.7611 |
| AVE(80-∞ , km/h) | 1.6001 | 2.4082 | 2.9308 | 2.9169 |


## 2.3 NMS 的选择对 Detection mAP 影响

针对实验一的检测模型，修改 NMS 方法进行测试（模型相同，都为 epoch_32.pth），实验了如下三种 NMS 方法，实验参数及测试结果如下：

| **NMS 方法** | **参数** | **VEHICLE_LEVEL_2/mAP** |**VEHICLE_LEVEL_2/mAPH** | **VEHICLE_LEVEL_2 Recall@0.95** |**PEDESTRIAN_LEVEL_2/mAP** | **PEDESTRIAN_LEVEL_2/mAPH** |**PEDESTRIAN_LEVEL_2** **Recall@0.95** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rotate NMS | score_threshold=0.2,nms_pre_max_size=1024,nms_post_max_size=256,nms_iou_threshold=0.2 | 0.73414207 | 0.72512126 | 0.6376742 | 0.68771183 | 0.6297276 | 0.30905744 |
| Circle NMS | score_threshold=0.2,nms_pre_max_size=1024,nms_post_max_size=256,nms_iou_threshold=0.2,**min_radius=[4, 0.175]** | 0.73500246 | 0.7259521 | 0.6198849 | 0.68583196 | 0.62698156 | 0.3695327 |
| ~~Max Pooling~~ | ~~kernel=3~~ | ~~0.46011257~~ | ~~0.45492992~~ | ~~0.0~~ | ~~0.5983365~~ | ~~0.5497276~~ | ~~0.26471835~~ |


# 三、初步结论

1、Simtrack 方法对于检测部分的网络结构改动不是很大，只是增加了 update branch 部分的学习难度，但是总体仍与 Centerpoint 架构一样，mAP 实验结果也得到了印证。

2、Simtrack 方法跟踪指标，略好于 Centerpoint 的贪心匹配及 KF 运动学模型匹配方法，其中，ID-switch 与 FN（误报）明显由于另外两个模型，从指标上看该方法有实用性。

3、从耗时分析，Simtrack 方法在 Detection 步骤，相比于 Centerpoint 没有增加任何额外的耗时。

**但是**其在跟踪过程中，需要根据上一帧的结果（BBoxes）重绘两张 heatmap 图（hm 图和 track_id 图），在 pytorch 的实现中，该步骤耗时较多，平均每帧要花费 20ms。因此，若要对该方案进行部署，后期需要重点优化该部分的耗时。

此外, 基于该模型进行推理, 必须采用 2-sweeps 的点云输入, 也会增加部署复杂度和推理耗时.

4、从可视化跟踪效果来看，Simtrack 与 Centerpoint 没有太大差距，后续可考虑加入卡尔曼滤波进行稳定。

5、从 Debug 角度来看，该方法有些融入 LSTM 的思想，在出现问题时不易复现和调试，可解释性较差。

6、目前存在一个问题, 即在 bev 视角下, 可能 heatmap 中同一个点出现多个目标的情况, 会导致相同的重复 ID, 后期必须针对该方法进行优化(考虑 rule-base 去重, 或者 circle-nms 等方法).
