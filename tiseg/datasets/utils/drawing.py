import random

import cv2
import matplotlib.pyplot as plt
import numpy as np


def colorize_seg_map(seg_map, palette=None):
    """using random rgb color to colorize segmentation map."""
    colorful_seg_map = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    id_list = list(np.unique(seg_map))

    if palette is None:
        palette = {}
        for id in id_list:
            color = [random.random() * 255 for i in range(3)]
            palette[id] = color

    for id in id_list:
        # ignore background
        if id == 0:
            continue
        colorful_seg_map[seg_map == id, :] = palette[id]

    return colorful_seg_map


def draw_semantic(save_folder, data_id, image, pred, label, edge_id=2):
    """draw semantic level picture with FP & FN."""

    plt.figure(figsize=(5 * 2, 5 * 2 + 3))

    # prediction drawing
    plt.subplot(221)
    plt.imshow(pred)
    plt.axis('off')
    plt.title('Prediction', fontsize=15, color='black')

    # ground truth drawing
    plt.subplot(222)
    plt.imshow(label)
    plt.axis('off')
    plt.title('Ground Truth', fontsize=15, color='black')

    # image drawing
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(223)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Image', fontsize=15, color='black')

    canvas = np.zeros((*pred.shape, 3), dtype=np.uint8)
    canvas[label > 0, :] = (255, 255, 2)
    canvas[canvas == edge_id] = 0
    canvas[(pred == 0) * (label > 0), :] = (2, 255, 255)
    canvas[(pred > 0) * (label == 0), :] = (255, 2, 255)
    plt.subplot(224)
    plt.imshow(canvas)
    plt.axis('off')
    plt.title('FN-FP-Ground Truth', fontsize=15, color='black')

    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [(255, 255, 2), (2, 255, 255), (255, 2, 255)]
    label_list = [
        'Ground Truth',
        'FN',
        'FP',
    ]
    for color, label in zip(colors, label_list):
        color = list(color)
        color = [x / 255 for x in color]
        plt.plot(0, 0, '-', color=tuple(color), label=label)
    plt.legend(loc='upper center', fontsize=9, bbox_to_anchor=(0.5, 0), ncol=3)

    # results visulization
    plt.tight_layout()
    plt.savefig(f'{save_folder}/{data_id}_semantic_compare.png', dpi=300)


def draw_instance(save_folder, data_id, pred_instance, label_instance):
    """draw instance level picture."""

    plt.figure(figsize=(5 * 2, 5))

    plt.subplot(121)
    plt.imshow(colorize_seg_map(pred_instance))
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(colorize_seg_map(label_instance))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_folder}/{data_id}_instance_compare.png', dpi=300)
