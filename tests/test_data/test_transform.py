import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from tiseg.datasets.pipelines import (CDNetLabelMake, DirectionMapCalculation,
                                      EdgeMapCalculation,
                                      InstanceMapCalculation,
                                      PointMapCalculation)
from tiseg.models.utils import generate_direction_differential_map


def test_split_label_calculation():
    pseudo_seg_map_path = osp.join(
        osp.dirname(__file__), '../data/nucleus_label_map.png')
    pseudo_seg_map = Image.open(pseudo_seg_map_path)
    pseudo_semantic_map = np.array(pseudo_seg_map)[:, :, 0]
    pseudo_semantic_map[pseudo_semantic_map == 255] = 1

    results_raw = {}
    results_raw['gt_semantic_map'] = pseudo_semantic_map
    results_raw['seg_fields'] = ['gt_semantic_map']

    # Test InstanceLabelCalculation
    transform = InstanceMapCalculation(remove_small_object=False)
    results_instance = transform(results_raw)
    pseudo_instance_map = results_instance['gt_instance_map']
    assert len(np.unique(pseudo_instance_map)) > 2
    plt.subplot(231)
    plt.imshow(pseudo_instance_map)

    # Test InstanceLabelCalculation with remove small object
    transform = InstanceMapCalculation(
        remove_small_object=True, object_small_size=10)
    results_instance = transform(results_raw)
    pseudo_instance_map = results_instance['gt_instance_map']
    assert len(np.unique(pseudo_instance_map)) > 2
    plt.subplot(232)
    plt.imshow(pseudo_instance_map)

    # Test EdgeLabelCalculation
    transform = EdgeMapCalculation(already_edge=False)
    results_edge = transform(results_raw)
    pseudo_semantic_map_edge = results_edge['gt_semantic_map_with_edge']
    assert len(np.unique(pseudo_semantic_map_edge)) == len(
        np.unique(pseudo_semantic_map)) + 1
    plt.subplot(233)
    plt.imshow(pseudo_semantic_map_edge)

    # Test PointMapCalculation
    transform = PointMapCalculation()
    results_point = transform(results_instance)
    pseudo_point_map = results_point['gt_point_map']
    assert pseudo_point_map.dtype == np.float32
    plt.subplot(234)
    plt.imshow(pseudo_point_map)

    # Test DirectionMapCalculation
    transform = DirectionMapCalculation(num_angle_types=8)
    results_direction = transform(results_point)
    pseudo_angle_map = results_direction['gt_angle_map']
    pseudo_direction_map = results_direction['gt_direction_map']
    plt.subplot(235)
    plt.imshow(pseudo_angle_map)
    plt.subplot(236)
    plt.imshow(pseudo_direction_map)
    plt.show()


def test_general_label_calculation():
    pseudo_seg_map_path = osp.join(
        osp.dirname(__file__), '../data/nucleus_label_map.png')
    pseudo_seg_map = Image.open(pseudo_seg_map_path)
    pseudo_semantic_map = np.array(pseudo_seg_map)[:, :, 0]
    pseudo_semantic_edge = np.array(pseudo_seg_map)[:, :, 1]
    pseudo_semantic_map[pseudo_semantic_map == 255] = 1
    pseudo_semantic_map[pseudo_semantic_edge == 255] = 2

    results_raw = {}
    results_raw['gt_semantic_map'] = pseudo_semantic_map
    results_raw['seg_fields'] = ['gt_semantic_map']

    transform = CDNetLabelMake()
    results = transform(results_raw)
    plt.subplot(221)
    plt.imshow(results['gt_semantic_map'])
    plt.subplot(222)
    plt.imshow(results['gt_semantic_map_with_edge'])
    plt.subplot(223)
    plt.imshow(results['gt_point_map'])
    plt.subplot(224)
    plt.imshow(results['gt_direction_map'])
    plt.show()

    # Test DirectionDifferentialMap
    inference_direction_map = torch.from_numpy(
        results['gt_direction_map']).expand(1, -1, -1)
    ddm = generate_direction_differential_map(inference_direction_map, 9)
    plt.subplot(121)
    plt.imshow(ddm.numpy()[0])
    plt.subplot(122)
    temp = results['gt_semantic_map_with_edge']
    ddm = ddm.numpy()[0]
    canvas = np.zeros((*temp.shape, 3), dtype=np.uint8)
    canvas[temp == 2, :] = (255, 255, 2)
    canvas[ddm > 0, :] = (2, 255, 255)
    plt.imshow(canvas)
    plt.show()
