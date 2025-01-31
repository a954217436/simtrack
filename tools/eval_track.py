import os, glob
import pickle
import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.python import tracking_metrics
from waymo_open_dataset.protos import metrics_pb2

"""
请注意：如果加载 gt 为 annos/seq_x_frame_xx.pkl, 里面的 "id" 并不是 track_id, 应是 "name"  ！！！

    python tools/eval_track.py

"""


class TrackinMetricsEstimatorTest(tf.test.TestCase):
    def _BuildConfig(self, additional_config_str=''):
        """Builds a metrics config."""
        config = metrics_pb2.Config()

        # OBJECT_TYPE adds 4 breakdowns
        # RANGE adds 12
        self.config_text = """
        num_desired_score_cutoffs: 2
        breakdown_generator_ids: OBJECT_TYPE
        difficulties {
            levels: LEVEL_1
        }
        breakdown_generator_ids: RANGE
        difficulties {
            levels: LEVEL_1
        }
        matcher_type: TYPE_HUNGARIAN
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        box_type: TYPE_2D
        score_cutoffs: 0.5
        score_cutoffs: 0.9
        """ + additional_config_str

        text_format.Merge(self.config_text, config)
        return config

    def _BuildGraph(self, graph):
        with graph.as_default():
            self._prediction_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._prediction_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._prediction_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self._prediction_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._prediction_sequence_id = tf.compat.v1.placeholder(dtype=tf.string)
            self._prediction_object_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._ground_truth_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._ground_truth_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._ground_truth_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._ground_truth_sequence_id = tf.compat.v1.placeholder(dtype=tf.string)
            self._ground_truth_object_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._ground_truth_difficulty = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._prediction_overlap_nlz = tf.compat.v1.placeholder(dtype=tf.bool)
            self._ground_truth_speed = tf.compat.v1.placeholder(dtype=tf.float32)

            metrics = tracking_metrics.get_tracking_metric_ops(
                config=self._BuildConfig(),
                prediction_bbox=self._prediction_bbox,
                prediction_type=self._prediction_type,
                prediction_score=self._prediction_score,
                prediction_frame_id=self._prediction_frame_id,
                prediction_sequence_id=self._prediction_sequence_id,
                prediction_object_id=self._prediction_object_id,
                ground_truth_bbox=self._ground_truth_bbox,
                ground_truth_type=self._ground_truth_type,
                ground_truth_frame_id=self._ground_truth_frame_id,
                ground_truth_sequence_id=self._ground_truth_sequence_id,
                ground_truth_object_id=self._ground_truth_object_id,
                ground_truth_difficulty=tf.ones_like(
                    self._ground_truth_frame_id, dtype=tf.uint8),
                prediction_overlap_nlz=tf.zeros_like(
                    self._prediction_frame_id, dtype=tf.bool),
                ground_truth_speed=self._ground_truth_speed)
            return metrics

    def _EvalUpdateOps(
        self,
        sess,
        graph,
        metrics,
        prediction_bbox,
        prediction_type,
        prediction_score,
        prediction_frame_id,
        prediction_sequence_id,
        prediction_object_id,
        ground_truth_bbox,
        ground_truth_type,
        ground_truth_frame_id,
        ground_truth_sequence_id,
        ground_truth_object_id,
        ground_truth_speed,
    ):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])],
            feed_dict={
                self._prediction_bbox: prediction_bbox,
                self._prediction_type: prediction_type,
                self._prediction_score: prediction_score,
                self._prediction_frame_id: prediction_frame_id,
                self._prediction_sequence_id: prediction_sequence_id,
                self._prediction_object_id: prediction_object_id,
                self._ground_truth_bbox: ground_truth_bbox,
                self._ground_truth_type: ground_truth_type,
                self._ground_truth_frame_id: ground_truth_frame_id,
                self._ground_truth_sequence_id: ground_truth_sequence_id,
                self._ground_truth_object_id: ground_truth_object_id,
                self._ground_truth_speed: ground_truth_speed,
            })

    def _EvalValueOps(self, sess, graph, metrics):
        # Get value_op from metrics dictionary.
        return {key: sess.run([value_op]) for key, (value_op, _) in metrics.items()}



# bboxes, types, frame_ids, sequence_ids, object_ids, scores, speed
def load_annos(pkl_path, frame_id=0, sequence_id='0'):
    # print("loading annos: ", pkl_path)
    annos = pickle.load(open(pkl_path, "rb"))
    objs = annos['objects']

    types       = np.array([a['label'] for a in objs])
    mask1 = types == 1
    mask2 = types == 2
    mask = np.logical_or(mask1, mask2)
    types = types[mask]

    nums = len(types)
    if (nums)==0:
        return None
    
    frame_ids = np.array([frame_id] * nums)
    sequence_ids = np.array([sequence_id] * nums)
    scores      = np.array([1.0] * nums)
    
    bboxes = np.array([a['box'] for a in objs])[mask]
    bboxes = bboxes[:, [0,1,2,3,4,5,8]]
    object_ids  = np.array([hash(a['name']) for a in objs])[mask]
    speed       = np.array([a['global_speed'] for a in objs])[mask]

    # print("gt: ", bboxes[:5])

    # object_uuids = [uuid_gen.get_uuid(int(b)) for b in object_ids[mask]]
    
    return bboxes, types, frame_ids, sequence_ids, object_ids, scores, speed


# bboxes, types, frame_ids, sequence_ids, object_ids, scores, speed
# WAYMO_GT_LABEL_DICT = {0:"cyc", 1:"car", 2:"ped", 3:"tra", 4:"other"}
# WAYMO_TK_LABEL_DICT = {0:'car', 1:'ped', 2:'cyc'}

LABEL_MAP = {0:1, 1:2, 2:0, 3:3, 4:4}
def load_txts(txt_path, frame_id=0, sequence_id='0', gt=True):
    # print("loading txt: ", txt_path)
    if not os.path.exists(txt_path):
        return None
    trks_npy = np.loadtxt(txt_path)
    if trks_npy.ndim == 1:
        return None
        
    nums = len(trks_npy)    
    frame_ids = np.array([frame_id] * nums)
    sequence_ids = [sequence_id] * nums
    
    object_ids = [int(b) for b in list(trks_npy[:, 0])]
    if gt:
        types = [int(a) for a in list(trks_npy[:, 1])]
    else:
        types = [LABEL_MAP[int(a)] for a in list(trks_npy[:, 1])]
    bboxes = trks_npy[:, 2:9]
    scores = trks_npy[:, -3]
    speed = trks_npy[:, -2:]
    
    # if not gt:
    #     bboxes = bboxes[:, [0, 1, 2, 4, 3, 5, 6]]
    #     bboxes[:, -1] = -bboxes[:, -1] - np.pi / 2 
    # print("tk: ", bboxes[:5])

    # object_uuids = [uuid_gen.get_uuid(int(b) + 9999) for b in object_ids]
    return bboxes, types, frame_ids, sequence_ids, object_ids, scores, speed


def calcMetrics(tk_path, anno_path = "/mnt/data/waymo_opensets/val/annos", seqs=range(202)):
    trackMetricsEst = TrackinMetricsEstimatorTest()
    graph = tf.Graph()
    metrics = trackMetricsEst._BuildGraph(graph)
    
    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.initializers.local_variables())
        
        for sss in seqs:
            pd_bboxs, pd_types, pd_frame_ids, pd_sequence_ids, pd_object_ids, pd_scores, pd_speeds = [],[],[],[],[],[],[]
            gt_bboxs, gt_types, gt_frame_ids, gt_sequence_ids, gt_object_ids, gt_scores, gt_speeds = [],[],[],[],[],[],[]
            frame_nums = len(glob.glob(anno_path + "/seq_%d_frame_*.pkl"%(sss)))
            print(sss, ": total frame_nums: ", frame_nums)
            for i in range(frame_nums):
                pd_infos = load_txts(tk_path + "/seq_%d_frame_%d.txt"%(sss,i), frame_id=i, sequence_id=str(sss), gt=False)                
                gt_infos = load_annos(anno_path + "/seq_%d_frame_%d.pkl"%(sss,i), frame_id=i, sequence_id=str(sss))
                
                if (pd_infos is None) or (gt_infos is None):
                    continue
                
                # print(pd_infos[6])
                # print("-"*50)
                # print(gt_infos[6])
                # import sys
                # sys.exit(-1)

                pd_bbox, pd_type, pd_frame_id, pd_sequence_id, pd_object_id, pd_score, pd_speed = pd_infos
                gt_bbox, gt_type, gt_frame_id, gt_sequence_id, gt_object_id, gt_score, gt_speed = gt_infos

                pd_bboxs.append(pd_bbox)
                pd_types.append(pd_type)
                pd_frame_ids.append(pd_frame_id)
                pd_sequence_ids.append(pd_sequence_id)
                pd_object_ids.append(pd_object_id)
                pd_scores.append(pd_score)
                pd_speeds.append(pd_speed)

                gt_bboxs.append(gt_bbox)
                gt_types.append(gt_type)
                gt_frame_ids.append(gt_frame_id)
                gt_sequence_ids.append(gt_sequence_id)
                gt_object_ids.append(gt_object_id)
                gt_scores.append(gt_score)
                gt_speeds.append(gt_speed)

            pd_bboxs = np.concatenate(pd_bboxs, axis=0)
            pd_types = np.concatenate(pd_types, axis=0)
            pd_frame_ids = np.concatenate(pd_frame_ids, axis=0)
            pd_sequence_ids = np.concatenate(pd_sequence_ids, axis=0)
            pd_object_ids = np.concatenate(pd_object_ids, axis=0)
            pd_scores = np.concatenate(pd_scores, axis=0)
            pd_speeds = np.concatenate(pd_speeds, axis=0)

            gt_bboxs        = np.concatenate(gt_bboxs, axis=0)
            gt_types        = np.concatenate(gt_types, axis=0)
            gt_frame_ids    = np.concatenate(gt_frame_ids, axis=0)
            gt_sequence_ids = np.concatenate(gt_sequence_ids, axis=0)
            gt_object_ids   = np.concatenate(gt_object_ids, axis=0)
            gt_scores       = np.concatenate(gt_scores, axis=0)
            gt_speeds       = np.concatenate(gt_speeds, axis=0)

            # print(pd_sequence_ids[-200:])
            # print(gt_sequence_ids[-200:])

            trackMetricsEst._EvalUpdateOps(sess, graph, metrics, pd_bboxs,
                                pd_types, pd_scores,
                                pd_frame_ids, pd_sequence_ids,
                                pd_object_ids, gt_bboxs,
                                gt_types, gt_frame_ids,
                                gt_sequence_ids, gt_object_ids,
                                gt_speeds)

        with tf.compat.v1.variable_scope('tracking_metrics', reuse=True):
            # Looking up an exisitng var to check that data is accumulated properly
            # in the variable.
            pd_frame_id_accumulated_var = tf.compat.v1.get_variable(
                'prediction_frame_id', dtype=tf.int64)
        
        pd_frame_id_accumulated = sess.run([pd_frame_id_accumulated_var])
        # print(pd_frame_id_accumulated)

        mot_metrics = trackMetricsEst._EvalValueOps(sess, graph, metrics)

        os.makedirs(tk_path + "/../metrics1/", exist_ok=True)
        with open(tk_path + "/../metrics1/mota_wod.txt", "w") as www:
            www.write("track result path : %s\n"%tk_path)
            www.write("track seqs : %s\n"%str(seqs))
            www.write("eval config : \n%s\n"%str(trackMetricsEst.config_text))

            for k,v in mot_metrics.items():
                print(k, v)
                www.write("%s : %s \n"%(k, v))
            

if __name__ == "__main__":
    # calcMetrics(tk_path="/home/zhanghao/code/GL/trackingwithvelo/py/data/output/all/22521_stablelose4_velo52_angle_120/txt", seqs=range(1))
    # calcMetrics(tk_path="/home/zhanghao/code/GL/trackingwithvelo/py/data/output/all/centerpoint_result/evaluation/txt")
    calcMetrics(tk_path="/home/zhanghao/code/master/4_TRACK/simtrack/work_dir/waymo/track_results_all_0.2/txt", seqs=range(202))
    # calcMetrics(tk_path="/home/zhanghao/code/GL/trackingwithvelo/py/data/output/all/gt_track/txt", seqs=range(202))

