import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor

# WAYMO_PATH = "/mnt/data/waymo_tiny_simtrack/"
WAYMO_PATH = "/mnt/data/waymo_opensets/"
norm_cfg = None

tasks = [
    dict(num_class=2, class_names=['VEHICLE', 'PEDESTRIAN'], stride=1),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))


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

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    nms=dict(
        nms_pre_max_size=4096,
        nms_post_max_size=300,  # 500
        nms_iou_threshold=0.6,  # 0.2  0.7
    ),
    score_threshold=0.3,  # 0.1
    pc_range=[-74.88, -74.88],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.32, 0.32],
    post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
    max_per_img=500,
)

# dataset settings
dataset_type = "WaymoDataset"
n_sweeps = 2
data_root = WAYMO_PATH

db_sampler = dict(
    type="GT-AUG",
    enable=True, 
    db_info_path= WAYMO_PATH + "/dbinfos_train_2sweeps_withvelo.pkl",
    sample_groups=[
        dict(VEHICLE=15),
        dict(PEDESTRIAN=10),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                VEHICLE=5,
                PEDESTRIAN=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    rate=1.0,
    gt_drop_percentage=0.5,
    gt_drop_max_keep_points=5,
    point_dim=6,    # 5 zhanghao
)

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

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,   #True
    remove_environment=False,
    remove_unknown_examples=False,
    class_names=class_names,
)

voxel_generator = dict(
    range=_pc_range,
    voxel_size=_voxel_size,
    max_points_in_voxel=20,  
    max_voxel_num=30000, 
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

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        # ann_file=train_anno,
        nsweeps=n_sweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        # ann_file=val_anno,
        test_mode=True,
        nsweeps=n_sweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        # ann_file=test_anno,
        nsweeps=n_sweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

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
