
"Do inference and evaluation offline"

"""
注意，该代码使用 val/annos/seq_xx_frame_yy.pkl 作为 gt, 需额外小心其坐标系定义与 Label 序号！！！

# 测试目录下的结果

1. 只测试不推理
    python tools/eval_det.py --save_path results/inference/waymo_pp_centernet_tracking_epoch_32 --noinfer --score_thres 0.1

2. 只推理不测试
    python tools/eval_det.py --config experiments/pointpillars/configs/waymo_pp_centernet_tracking.py --ckpt work_dir/waymo/epoch_32.pth --noeval

3. 推理并测试
    python tools/eval_det.py --config experiments/pointpillars/configs/waymo_pp_centernet_tracking.py --ckpt work_dir/waymo/epoch_32.pth

    CUDA_VISIBLE_DEVICES=1 python tools/eval_det.py \
        --config work_dirs/waymo_pp_centernet_tracking2/configs/waymo_pp_centernet_tracking2.py \
        --ckpt work_dirs/waymo_pp_centernet_tracking2/epoch_33.pth \
        --batch_size 16 \
        --num_worker 16
"""

import os
import numpy as np
import torch
import torch.multiprocessing as mp


from det3d.datasets import build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.trainer import load_checkpoint
import time 
from det3d.torchie.parallel import collate_kitti
from torch.utils.data import DataLoader


import argparse
from tqdm import tqdm
import pickle as pkl
from multiprocessing import Pool
#For Metics Computation
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2
ERROR = 1e-6

# bboxes, types, frame_ids, sequence_ids, object_ids, scores, speed
# WAYMO_GT_LABEL_DICT = {0:"cyc", 1:"car", 2:"ped", 3:"tra", 4:"other"}
# WAYMO_TK_LABEL_DICT = {0:'car', 1:'ped', 2:'cyc'}
LABEL_MAP = {0:1, 1:2, 2:0, 3:3, 4:4}

METRIC_CONFIG = """
        num_desired_score_cutoffs: 11
        breakdown_generator_ids: OBJECT_TYPE
        difficulties {
        }
        matcher_type: TYPE_HUNGARIAN
        iou_thresholds: 0.5
        iou_thresholds: 0.7
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        box_type: TYPE_2D
        """
# METRIC_CONFIG = """
#         num_desired_score_cutoffs: 11
#         breakdown_generator_ids: OBJECT_TYPE
#         difficulties {
#         levels: 1
#         levels: 2
#         }
#         matcher_type: TYPE_HUNGARIAN
#         iou_thresholds: 0.5
#         iou_thresholds: 0.7
#         iou_thresholds: 0.5
#         iou_thresholds: 0.5
#         iou_thresholds: 0.5
#         box_type: TYPE_2D
#         """

RECALL_AT = 0.95

class DetectionMetricsEstimatorTest(tf.test.TestCase):
    def _BuildConfig(self):
        config = metrics_pb2.Config()
        config_text = METRIC_CONFIG
        text_format.Merge(config_text, config)
        return config

    def _BuildGraph(self, graph):
        with graph.as_default():
            self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)
      
            metrics = detection_metrics.get_detection_metric_ops(
                config=self._BuildConfig(),
                prediction_frame_id=self._pd_frame_id,
                prediction_bbox=self._pd_bbox,
                prediction_type=self._pd_type,
                prediction_score=self._pd_score,
                prediction_overlap_nlz=tf.zeros_like(
                    self._pd_frame_id, dtype=tf.bool),
                ground_truth_bbox=self._gt_bbox,
                ground_truth_type=self._gt_type,
                ground_truth_frame_id=self._gt_frame_id,
                ground_truth_difficulty=tf.ones_like(
                    self._gt_frame_id, dtype=tf.uint8),
                # recall_at_precision=0.95,
                recall_at_precision=RECALL_AT,
            )
        return metrics

    def _EvalUpdateOps(
      self,
      sess,
      graph,
      metrics,
      prediction_frame_id,
      prediction_bbox,
      prediction_type,
      prediction_score,
      ground_truth_frame_id,
      ground_truth_bbox,
      ground_truth_type,
      ):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])],
            feed_dict={
                self._pd_bbox: prediction_bbox,
                self._pd_frame_id: prediction_frame_id,
                self._pd_type: prediction_type,
                self._pd_score: prediction_score,
                self._gt_bbox: ground_truth_bbox,
                self._gt_type: ground_truth_type,
                self._gt_frame_id: ground_truth_frame_id,
            })

    def _EvalValueOps(self, sess, graph, metrics):
        return {item[0]: sess.run([item[1][0]]) for item in metrics.items()}


def example_to_device(example, device=None, non_blocking=False) -> dict:
    assert device is not None
    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels"]:    
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            # "points",
            "num_voxels",
            "cyv_voxels",
            "cyv_num_voxels",
            "cyv_coordinates",
            "cyv_num_points"]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                # calib[k1] = torch.tensor(v1, dtype=dtype, device=device)
                calib[k1] = torch.tensor(v1).to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        elif k == "points":
            # zhanghao: 务必注意 points 的 type,如果不是列表, 不能用遍历方法, 否则速度特别慢！！！
            if type(v) == list:
                example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
            else:
                example_torch[k] = v.to(device, non_blocking=non_blocking)
        else:
            example_torch[k] = v
    return example_torch


def pkl_read(p):
	data = pkl.load(open(p, 'rb'))
	return data


def load_anno_gt(anno_path):
    all_data = pkl_read(anno_path)

    gt_boxes = np.array([ann['box'] for ann in all_data['objects']]).reshape(-1, 9)
    gt_classes = np.array([ann['label'] for ann in all_data['objects']])
    return gt_boxes, gt_classes


def classname2label(data):
    return np.array([CLASSNAME2LABEL[x] for x in data])


def dict_to_cpu(data):
    for k,v in data.items():
        if hasattr(v,"cpu"):
            data[k] = v.cpu().numpy()
    return data


def save_predictions(worker_id, data_loader, model, gpu_device, args):
    with torch.no_grad() :
        for idx, data_batch in tqdm(enumerate(data_loader)):
        # # for idx in tqdm(range(data_loader.__len_())):
        #     if idx % args.num_worker != worker_id:
        #         continue
            # data_batch = data_loader.__getitem__(idx)
            example = example_to_device(data_batch, device=gpu_device, non_blocking=True)
            rets = model(example, return_loss=False)

            for i, ret in enumerate(rets):
                ret = rets[i]
                ret = dict_to_cpu(ret)
                token = ret['metadata']['token']
                # 保存 pickle 文件
                sp = os.path.join(args.save_path, token)
                pkl.dump(ret, open(sp, 'wb'))


def run_inference(args):
    gpu_device = torch.device("cuda")

    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    ckpt_path = cfg.PRETRAINED if args.ckpt is None else args.ckpt
    checkpoint = load_checkpoint(model, ckpt_path, map_location="cpu")
    model.eval()
    model.cuda()
    print("Model has been loaded from %s"  %  ckpt_path)

    dataset = build_dataset(cfg.data.val)
    data_loader = DataLoader(
        dataset,batch_size=args.batch_size,
        sampler=None,
        shuffle=False,
        num_workers=args.num_worker,
        collate_fn=collate_kitti,
        pin_memory=True,
    )
    # data_iter = iter(data_loader)
    # data_batch = next(data_iter)
    print(cfg.data.val)
    print("Dataloader size : ", len(data_loader))
    print("Start Running Inference . . .")

    save_predictions(0, data_loader, model, gpu_device, args)


def compute_detection_metrics(args):
    # val_infos = pkl_read(args.info_path)
    sample_detection_metrics = DetectionMetricsEstimatorTest()
    graph = tf.Graph()
    metrics = sample_detection_metrics._BuildGraph(graph)
    # with sample_detection_metrics.test_session(graph=graph) as sess:

    det_pkls = os.listdir(args.save_path)

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.initializers.local_variables())
        pd_boxes, pd_classes, pd_frame_ids, pd_scores = [],[],[],[]
        gt_boxes, gt_classes, gt_frame_ids = [],[],[]

        # for idx, info in tqdm(enumerate(val_infos)):
        for idx, det_pkl_token in tqdm(enumerate(det_pkls)):
            pd_file = os.path.join(args.save_path, det_pkl_token)
            gt_file = os.path.join(args.anno_path, det_pkl_token)
            if not os.path.exists(gt_file):
                print("Anno file does not exist: ", gt_file)
                continue

            pred_dict = pkl_read(pd_file)
            scores = pred_dict['scores']
            labels = np.array([LABEL_MAP[int(i)] for i in pred_dict['label_preds']])  
            selected_indexs = np.where(scores >= args.score_thres)[0]
            if len(selected_indexs)==0:
                continue

            pd_boxes.append(pred_dict['box3d_lidar'][selected_indexs][:,[0,1,2,3,4,5,-1]])
            pd_classes.append(labels[selected_indexs])
            pd_scores.append(scores[selected_indexs])
            pd_frame_ids.append(np.ones(len(selected_indexs)) * idx)

            # print(pd_classes)
            # print(pd_boxes)
            # print("-+"*50)
            
            gt_box9ds, gt_labels = load_anno_gt(gt_file)
            gt_boxes.append(gt_box9ds[:, [0,1,2,3,4,5,-1]])
            gt_classes.append(gt_labels)
            gt_frame_ids.append(np.ones(len(gt_box9ds)) * idx)

            # print(gt_classes)
            # print(gt_boxes)
            # c=1/0

            if idx % 100 ==0 or idx == len(det_pkls) - 1:
                pred_boxes = np.concatenate(pd_boxes, axis=0)
                pred_classes = np.concatenate(pd_classes, axis = 0)
                pred_scores = np.concatenate(pd_scores, axis=0)
                pred_frame_ids = np.concatenate(pd_frame_ids)

                gt_boxes = np.concatenate(gt_boxes)
                gt_classes = np.concatenate(gt_classes)
                gt_frame_ids = np.concatenate(gt_frame_ids)


                sample_detection_metrics._EvalUpdateOps(sess, graph, metrics, pred_frame_ids, pred_boxes, pred_classes,\
                            pred_scores, gt_frame_ids, gt_boxes, gt_classes)
                pd_boxes , pd_classes , pd_scores, pd_frame_ids, gt_frame_ids, gt_boxes, gt_classes = [],[],[],[],[],[],[]
          # Looking up an exisitng var to check that data is accumulated properly
          # in the variable
        aps = sample_detection_metrics._EvalValueOps(sess, graph, metrics)
        print("EVALUATION RESULTS : \n")

        if not args.nosave_txt:
            txt_save_path = "results/eval/%s.txt"%(time.strftime("%Y%m%d_%H%M%S"))
            with open(txt_save_path, "a") as tt:
                tt.writelines(METRIC_CONFIG)
                tt.writelines("thresh = " + str(args.score_thres)+"\n")
                tt.writelines(args.save_path+"\n\n")

        for k,v in aps.items():
            print(k, v)
            if not args.nosave_txt:
                with open(txt_save_path, "a") as tt:
                    tt.write(str(k))
                    tt.write(str(v)+"\n")
        return aps


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a detector")
    parser.add_argument("--config", help="train config file path",type=str, default='configs/waymo/pp/waymo_centerpoint_pp_two_pfn_stride1_3x.py')
    parser.add_argument("--ckpt", help="ckpt of the model",type = str,  default = "epoch_8.pth")
    parser.add_argument("--save_path", help="the dir to save outputs",type = str, default = None)
    # parser.add_argument("--info_path", help="the path to gt infos",type = str, default = "/mnt/data/waymo_opensets/infos_val_01sweeps_filter_zero_gt.pkl")
    parser.add_argument("--anno_path", help="the path to gt infos",type = str, default = "/data/waymo_opensets/val/annos/")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size to infers")
    parser.add_argument("--num_worker", type=int, default=16, help="num workers to infers")
    parser.add_argument("--score_thres", type=float, default=0.2, help="score thresholds")
    parser.add_argument("--noinfer", action="store_true", default=False, help="whether to inference")
    parser.add_argument("--noeval", action="store_true", default=False, help="whether to evaluate")
    parser.add_argument("--nosave_txt", action="store_true", default=False, help="whether to save in txt")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.makedirs("results/eval/", exist_ok=True)

    args = parse_args()
    if args.save_path is None:
        args.save_path = "results/inference/%s_%s"%(args.config.split("/")[-1][:-3], args.ckpt.split("/")[-1][:-4])
        os.makedirs(args.save_path, exist_ok=True)

    if not args.noinfer:
        run_inference(args)
        print("Finished inference, saving in : ", args.save_path)
    

    if not args.noeval:
        print("Computing 3D Detection Metrics in : ", args.save_path)
        compute_detection_metrics(args)

