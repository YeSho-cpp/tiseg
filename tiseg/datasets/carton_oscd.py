import os.path as osp
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology
from skimage.morphology import remove_small_objects
from torch.utils.data import Dataset

from tiseg.utils.evaluation.metrics import (accuracy, aggregated_jaccard_index,
                                            dice_similarity_coefficient,
                                            precision_recall)
from .builder import DATASETS
from .pipelines import Compose
from .utils import re_instance


@DATASETS.register_module()
class CartonOSCDDataset(Dataset):
    """Carton Custom Foundation Segmentation Dataset.

    Although, this dataset is a instance segmentation task, this dataset also
    support a three class semantic segmentation task (Background, Carton,
    Edge).

    related suffix:
        "_semantic_with_edge.png": three class semantic map with edge.
        "_polygon.json": overlapping instance polygons.
        "_instance.npy": instance level map.
    """

    CLASSES = ('background', 'carton', 'edge')

    EDGE_ID = 2

    PALETTE = [[0, 0, 0], [255, 2, 255], [2, 255, 255]]

    def __init__(self,
                 pipeline,
                 img_dir,
                 ann_dir,
                 data_root=None,
                 img_suffix='.jpg',
                 ann_suffix='_semantic_with_edge.png',
                 test_mode=False,
                 split=None):

        # semantic level input or instance level input
        assert ann_suffix in ['_semantic_with_edge.png', '_instance.npy']
        if ann_suffix == '_semantic_with_edge.png':
            self.input_level = 'semantic_with_edge'
        elif ann_suffix == '_instance.npy':
            self.input_level = 'instance'

        self.pipeline = Compose(pipeline)

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.data_root = data_root

        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix

        self.test_mode = test_mode
        self.split = split

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        self.data_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                                self.ann_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def __getitem__(self, index):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        if self.test_mode:
            return self.prepare_test_data(index)
        else:
            return self.prepare_train_data(index)

    def prepare_test_data(self, index):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """
        data_info = self.data_infos[index]
        results = self.pre_pipeline(data_info)
        return self.pipeline(results)

    def prepare_train_data(self, index):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        data_info = self.data_infos[index]
        results = self.pre_pipeline(data_info)
        return self.pipeline(results)

    def pre_pipeline(self, data_info):
        """Prepare results dict for pipeline."""
        results = {}
        results['img_info'] = {}
        results['ann_info'] = {}

        # path retrieval
        results['img_info']['img_name'] = data_info['img_name']
        results['img_info']['img_dir'] = self.img_dir
        results['ann_info']['ann_name'] = data_info['ann_name']
        results['ann_info']['ann_dir'] = self.ann_dir

        # build seg fileds
        results['seg_fields'] = []

        return results

    def load_annotations(self, img_dir, img_suffix, ann_suffix, split=None):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory.
            ann_dir (str): Path to annotation directory.
            img_suffix (str): Suffix of images.
            ann_suffix (str): Suffix of segmentation maps.
            split (str | None): Split txt file. If split is specified, only
                file with suffix in the splits will be loaded.

        Returns:
            list[dict]: All data info of dataset, data info contains image,
                segmentation map.
        """
        data_infos = []
        if split is not None:
            with open(split, 'r') as fp:
                for line in fp.readlines():
                    img_id = line.strip()
                    image_name = img_id + img_suffix
                    ann_name = img_id + ann_suffix
                    data_info = dict(img_name=image_name, ann_name=ann_name)
                    data_infos.append(data_info)
        else:
            for img_name in mmcv.scandir(img_dir, img_suffix, recursive=True):
                ann_name = img_name.replace(img_suffix, ann_suffix)
                data_info = dict(img_name=img_name, ann_name=ann_name)
                data_infos.append(data_info)

        return data_infos

    def get_gt_seg_maps(self):
        """Ground Truth maps generator."""
        for data_info in self.data_infos:
            seg_map = osp.join(self.ann_dir, data_info['ann_name'])
            gt_seg_map = mmcv.imread(
                seg_map, flag='unchanged', backend='pillow')
            yield gt_seg_map

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.
            show_semantic (bool): Illustrate semantic level prediction &
                ground truth. Default: False
            show_instance (bool): Illustrate instance level prediction &
                ground truth. Default: False
            show_folder (str | None, optional): The folder path of
                illustration. Default: None

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = osp.join(self.ann_dir,
                               self.data_infos[index]['ann_name'])
            # (after pre_eval input process) requires 4 type seg_map:
            # 1. seg_map_semantic_edge (semantic map ground truth with edge)
            # 2. seg_map_segamtic (raw semantic map ground truth)
            # 3. seg_map_edge (semantic level edge ground truth)
            # 4. seg_map_instance (instance level ground truth)
            if self.input_level == 'semantic_with_edge':
                # semantic edge level label make
                seg_map_semantic_edge = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
                seg_map_edge = (seg_map_semantic_edge == self.EDGE_ID).astype(
                    np.uint8)
                # ground truth of semantic level
                seg_map_semantic = seg_map.replace('_semantic_with_edge.png',
                                                   '_semantic.png')
                seg_map_semantic = mmcv.imread(
                    seg_map_semantic, flag='unchanged', backend='pillow')
                # instance level label make
                seg_map_instance = seg_map.replace('_semantic_with_edge.png',
                                                   '_instance.npy')
                seg_map_instance = np.load(seg_map_instance)
                seg_map_instance = re_instance(seg_map_instance)

            # metric calculation post process codes:
            pred_semantic_edge = pred
            # edge id is 2
            pred_edge = (pred_semantic_edge == self.EDGE_ID).astype(np.uint8)
            pred_semantic = pred.copy()
            pred_semantic[pred_edge > 0] = 0

            # model-agnostic post process operations
            pred_semantic, pred_instance = self.model_agnostic_postprocess(
                pred_semantic)

            # semantic metric calculation (remove background class)
            # [1] will remove background class.
            precision_metric, recall_metric = precision_recall(
                pred_semantic, seg_map_semantic, 2)
            precision_metric = precision_metric[1]
            recall_metric = recall_metric[1]
            dice_metric = dice_similarity_coefficient(pred_semantic,
                                                      seg_map_semantic, 2)[1]
            acc_metric = accuracy(pred_semantic, seg_map_semantic, 2)[1]

            edge_precision_metric, edge_recall_metric = \
                precision_recall(pred_edge, seg_map_edge, 2)
            edge_precision_metric = edge_precision_metric[1]
            edge_recall_metric = edge_recall_metric[1]
            edge_dice_metric = dice_similarity_coefficient(
                pred_edge, seg_map_edge, 2)[1]

            # instance metric calculation
            aji_metric = aggregated_jaccard_index(
                pred_instance, seg_map_instance, is_semantic=False)

            single_loop_results = dict(
                Aji=aji_metric,
                Dice=dice_metric,
                Accuracy=acc_metric,
                Recall=recall_metric,
                Precision=precision_metric,
                edge_Dice=edge_dice_metric,
                edge_Recall=edge_recall_metric,
                edge_Precision=edge_precision_metric)
            pre_eval_results.append(single_loop_results)

        return pre_eval_results

    def model_agnostic_postprocess(self, pred):
        """model free post-process for both instance-level & semantic-level."""
        id_list = list(np.unique(pred))
        pred_canvas = np.zeros_like(pred).astype(np.uint8)
        for id in id_list:
            if id == 0:
                continue
            id_mask = pred == id
            # fill instance holes
            id_mask = binary_fill_holes(id_mask)
            # remove small instance
            id_mask = remove_small_objects(id_mask, 20)
            id_mask = id_mask.astype(np.uint8)
            pred_canvas[id_mask > 0] = id
        pred_semantic = pred_canvas.copy()
        pred_semantic = morphology.dilation(
            pred_semantic, selem=morphology.disk(2))

        # instance process & dilation
        pred_instance = measure.label(pred_canvas > 0)
        # if re_edge=True, dilation pixel length should be 2
        pred_instance = morphology.dilation(
            pred_instance, selem=morphology.disk(2))

        return pred_semantic, pred_instance

    def evaluate(self,
                 results,
                 metric='all',
                 logger=None,
                 dump_path=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            processor (object): The result processor.
            metric (str | list[str]): Metrics to be evaluated. 'Aji',
                'Dice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            dump_path (str | None, optional): The dump path of each item
                evaluation results. Default: None

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        if 'all' in metric:
            metric = ['IoU', 'Dice', 'Precision', 'Recall', 'Accuracy']
        allowed_metrics = ['IoU', 'Dice', 'Precision', 'Recall', 'Accuracy']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        ret_metrics = {}
        # list to dict
        for result in results:
            for key, value in result.items():
                if key not in ret_metrics:
                    ret_metrics[key] = [value]
                else:
                    ret_metrics[key].append(value)

        # calculate average metric
        for key in ret_metrics.keys():
            # XXX: Using average value may have lower metric value than using
            # confused matrix.
            average_value = sum(ret_metrics[key]) / len(ret_metrics[key])

            ret_metrics[key] = average_value

        # for logger
        ret_metrics_items = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_items.update({'name': 'Average'})
        ret_metrics_items.move_to_end('name', last=False)
        items_table_data = PrettyTable()
        for key, val in ret_metrics_items.items():
            items_table_data.add_column(key, [val])

        print_log('Total:', logger)
        print_log('\n' + items_table_data.get_string(), logger=logger)

        # dump to txt
        if dump_path is not None:
            fp = open(f'{dump_path}', 'w')
            fp.write(items_table_data.get_string())

        eval_results = {}
        # average results
        for key, val in ret_metrics.items():
            if key != 'Aji' and 'edge' not in key:
                key = 'm' + key
            eval_results.update({key: val})

        # This ret value is used for eval hook. Eval hook will add these
        # evaluation info to runner.log_buffer.output. Then when the
        # TextLoggerHook is called, the evaluation info will output to logger.
        return eval_results
