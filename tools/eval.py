
"Do inference and evaluation offline"

"""
# 测试目录下的结果
python tools/eval.py \
    --noinference \
    --save_path save_dir_tmp

# 推理并测试
python tools/eval.py \
    --config work_dir/waymo/waymo_pp_centernet_tracking.py \
    --ckpt work_dir/waymo/epoch_7.pth
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

CLASSNAME2LABEL = {"VEHICLE" : 0, "PEDESTRIAN" : 1, "CYCLIST" : 2, "SIGN" : 3, "UNKNOWN" : 3}
BATCH_SIZE = 2
SCORE_THRE = 0.2

# METRIC_CONFIG = """
#         num_desired_score_cutoffs: 11
#         breakdown_generator_ids: OBJECT_TYPE
#         difficulties {
#         }
#         matcher_type: TYPE_HUNGARIAN
#         iou_thresholds: 0.5
#         iou_thresholds: 0.5
#         iou_thresholds: 0.5
#         iou_thresholds: 0.5
#         iou_thresholds: 0.5
#         box_type: TYPE_3D
#         """
METRIC_CONFIG = """
        num_desired_score_cutoffs: 11
        breakdown_generator_ids: OBJECT_TYPE
        difficulties {
        levels: 1
        levels: 2
        }
        matcher_type: TYPE_HUNGARIAN
        iou_thresholds: 0.5
        iou_thresholds: 0.7
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        box_type: TYPE_3D
        """

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
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "points"]:
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
        else:
            example_torch[k] = v
    return example_torch


def pkl_read(p):
	data = pkl.load(open(p,'rb'))
	return data


def classname2label(data):
    return np.array([CLASSNAME2LABEL[x] for x in data])


def dict_to_cpu(data):
    for k,v in data.items():
        if hasattr(v,"cpu"):
            data[k] = v.cpu().numpy()
    return data


def save_predictions(worker_id, data_loader,model,gpu_device, args):
    with torch.no_grad() :
        for idx, data_batch in tqdm(enumerate(data_loader)):
        # for idx in tqdm(range(data_loader.__len_())):
            if  idx % args.num_worker != worker_id:
                continue

            # data_batch = data_loader.__getitem__(idx)
            example = example_to_device(data_batch,device=gpu_device, non_blocking=True)
            rets = model(example,return_loss=False)
            for i, ret in enumerate(rets):
                ret = rets[i]
                ret = dict_to_cpu(ret)
                token = ret['metadata']['token']
                sp = os.path.join(args.save_path,token)
                pkl.dump(ret, open(sp,'wb'))


def run_inference(args):
    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    print(cfg.data.val)
    dataset = build_dataset(cfg.data.val)
    data_loader = DataLoader(
        dataset,batch_size=BATCH_SIZE,
        sampler=None,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_kitti,
        pin_memory=False,
    )
    # data_iter = iter(data_loader)
    # data_batch = next(data_iter)
    ckpt_path = cfg.PRETRAINED if args.ckpt is None else args.ckpt
    checkpoint = load_checkpoint(model, ckpt_path, map_location="cpu")
    print("Model has been loaded from %s"  %  ckpt_path)
    model.eval()
    gpu_device = torch.device("cuda")
    # (example["voxels"],example["num_points"],example["coordinates"]),\
    #                             gpu_device, non_blocking=False)
    model.cuda()
    print("Running  Inference . . .")
    if args.num_worker == 1:
        save_predictions(0, data_loader, model, gpu_device, args)
    else:
        mp.spawn(save_predictions, nprocs = args.num_worker, args = (data_loader, model,gpu_device,  args))
        # pool = Pool(args.num_worker)
        # try:
        #     for worker_id in range(args.num_worker):
        #         pool.apply_async(save_predictions, (worker_id,data_loader,model,args))
        #     pool.close()
        #     pool.join()
        # except Exception as e:
        #     print(e)
        # finally:
        #     print("Success finished saveing detection outputs !")


def compute_detection_metrics(args):
    val_infos = pkl_read(args.info_path)
    sample_detection_metrics = DetectionMetricsEstimatorTest()
    graph = tf.Graph()
    metrics = sample_detection_metrics._BuildGraph(graph)
    # with sample_detection_metrics.test_session(graph=graph) as sess:
    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.initializers.local_variables())
        pred_boxes , pred_classes , pred_scores, pred_frame_ids,gt_frame_ids, gt_boxes, gt_classes = [],[],[],[],[],[],[]
        for idx, info in tqdm(enumerate(val_infos)):
            pred_name = os.path.basename(info['path'])
            pred_file = os.path.join(args.save_path,pred_name)
            if not os.path.exists(pred_file):
                continue

            pred_dict = pkl_read(pred_file)
            scores = pred_dict['scores']
            selected_indexs = np.where(scores >= SCORE_THRE)[0]
            if len(selected_indexs)==0:
                continue
            # print("pred_boxes : \n",scores )
            # print("gt_boxes : \n",classname2label(info['gt_names']) )

            pred_boxes.append(pred_dict['box3d_lidar'][selected_indexs][:,[0,1,2,3,4,5,-1]])
            pred_classes.append(pred_dict['label_preds'][selected_indexs])
            pred_scores.append(scores[selected_indexs])
            pred_frame_ids.append(np.ones(len(selected_indexs)) * idx)

            # omit vel_x and vel_y
            # gt_boxes.append(info['gt_boxes'][:,[0,1,2,3,4,5,8]])
            
            # # 如果用的是 infos_val_01sweeps_filter_zero_gt.pkl, 其box定义不一样！！！
            boxes_gt_simtrack = info['gt_boxes'][:,[0,1,2,4,3,5,8]]
            boxes_gt_simtrack[:,-1] = -np.pi / 2.0 - boxes_gt_simtrack[:,-1]
            gt_boxes.append(boxes_gt_simtrack)

            gt_classes.append(classname2label(info['gt_names']))
            gt_frame_ids.append(np.ones(len(info['gt_boxes'])) * idx)

            if idx % 100 ==0 or idx == len(val_infos)-1:
                pred_boxes = np.concatenate(pred_boxes,axis=0)
                pred_classes = np.concatenate(pred_classes,axis = 0)
                pred_scores = np.concatenate(pred_scores,axis=0)
                pred_frame_ids = np.concatenate(pred_frame_ids)
                gt_boxes = np.concatenate(gt_boxes)
                gt_classes = np.concatenate(gt_classes)
                gt_frame_ids = np.concatenate(gt_frame_ids)


                sample_detection_metrics._EvalUpdateOps(sess, graph, metrics, pred_frame_ids, pred_boxes, pred_classes,\
                            pred_scores, gt_frame_ids, gt_boxes, gt_classes)
                pred_boxes , pred_classes , pred_scores, pred_frame_ids, gt_frame_ids, gt_boxes, gt_classes = [],[],[],[],[],[],[]
          # Looking up an exisitng var to check that data is accumulated properly
          # in the variable
        aps = sample_detection_metrics._EvalValueOps(sess, graph, metrics)
        print("EVALUATION RESULTS : \n")

        if not args.nosave_txt:
            txt_save_path = "results_eval/%s.txt"%(time.strftime("%Y%m%d_%H%M%S"))
            with open(txt_save_path, "a") as tt:
                tt.writelines(METRIC_CONFIG)
                tt.writelines("thresh = " + str(SCORE_THRE)+"\n")
                tt.writelines(args.save_path+"\n\n")

        for k,v in aps.items():
            print(k, v)
            if not args.nosave_txt:
                with open(txt_save_path, "a") as tt:
                    tt.write(str(k))
                    tt.write(str(v)+"\n")
        return aps


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--config", help="train config file path",type=str, default='configs/waymo/pp/waymo_centerpoint_pp_two_pfn_stride1_3x.py')
    parser.add_argument("--ckpt", help="ckpt of the model",type = str,  default = "epoch_8.pth")
    parser.add_argument("--save_path", help="the dir to save outputs",type = str, default = None)
    parser.add_argument("--info_path", help="the path to gt infos",type = str, default = "/mnt/data/waymo_opensets/infos_val_01sweeps_filter_zero_gt.pkl")
    parser.add_argument("--num_worker",type=int, default=1, help="num workers to infers")
    parser.add_argument("--noinference", action="store_true", default=False, help="whether to do inference")
    parser.add_argument("--nosave_txt", action="store_true", default=False, help="whether to save in txt")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.save_path is None:
        args.save_path = "results_inference/%s_%s"%(args.config.split("/")[-1][:-3], args.ckpt.split("/")[-1][:-4])
        os.makedirs(args.save_path, exist_ok=True)

    if not args.noinference:
        run_inference(args)

    print("Computing 3D Detection Metrics . . .")
    print("DT-folder: ", args.save_path)
    compute_detection_metrics(args)

