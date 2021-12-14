import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tiseg.utils import resize
from tiseg.utils.evaluation.metrics import aggregated_jaccard_index
from ..builder import SEGMENTORS
from ..losses import MultiClassDiceLoss, miou, tiou
from ..utils import generate_direction_differential_map
from .base import BaseSegmentor
from .try_cdnet_backbone import Unet


@SEGMENTORS.register_module()
class TryCDNetSegmentor(BaseSegmentor):
    """Base class for segmentors."""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(TryCDNetSegmentor, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.num_angles = 8

        self.unet = Unet()

    def calculate(self, img):
        mask_logit, dir_logit, point_logit = self.unet(img)

        mask_logit = resize(input=mask_logit, size=img.shape[2:], mode='bilinear', align_corners=False)
        dir_logit = resize(input=dir_logit, size=img.shape[2:], mode='bilinear', align_corners=False)
        point_logit = resize(input=point_logit, size=img.shape[2:], mode='bilinear', align_corners=False)

        return mask_logit, dir_logit, point_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            mask_logit, dir_logit, point_logit = self.calculate(data['img'])

            assert label is not None
            mask_gt = label['sem_gt_w_bound']
            point_gt = label['point_gt']
            dir_gt = label['dir_gt']

            loss = dict()
            mask_logit = resize(input=mask_logit, size=mask_gt.shape[2:])
            dir_logit = resize(input=dir_logit, size=dir_gt.shape[2:])
            point_logit = resize(input=point_logit, size=point_gt.shape[2:])

            mask_gt = mask_gt.squeeze(1)
            dir_gt = dir_gt.squeeze(1)

            # TODO: Conside to remove some edge loss value.
            # mask branch loss calculation
            mask_loss = self._mask_loss(mask_logit, mask_gt)
            loss.update(mask_loss)
            # direction branch loss calculation
            dir_loss = self._dir_loss(dir_logit, dir_gt)
            loss.update(dir_loss)
            # point branch loss calculation
            point_loss = self._point_loss(point_logit, point_gt)
            loss.update(point_loss)

            # calculate training metric
            training_metric_dict = self._training_metric(mask_logit, dir_logit, point_logit, mask_gt, dir_gt, point_gt)
            loss.update(training_metric_dict)
            return loss
        else:
            assert self.test_cfg is not None
            # NOTE: only support batch size = 1 now.
            seg_logit = self.inference(data['img'], metas[0], True)
            seg_pred = seg_logit.argmax(dim=1)
            # Extract inside class
            seg_pred = seg_pred.cpu().numpy()
            # unravel batch dim
            seg_pred = list(seg_pred)
            return seg_pred

    def inference(self, img, meta, rescale):
        """Inference with split/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            meta (dict): Image info dict.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['split', 'whole']

        self.rotate_degrees = self.test_cfg.get('rotate_degrees', [0])
        self.flip_directions = self.test_cfg.get('flip_directions', ['none'])
        sem_logit_list = []
        dir_logit_list = []
        point_logit_list = []
        img_ = img
        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                img = self.tta_transform(img_, rotate_degree, flip_direction)

                # inference
                if self.test_cfg.mode == 'split':
                    sem_logit, dir_logit, point_logit = self.split_inference(img, meta, rescale)
                else:
                    sem_logit, dir_logit, point_logit = self.whole_inference(img, meta, rescale)

                sem_logit = self.reverse_tta_transform(sem_logit, rotate_degree, flip_direction)
                dir_logit = self.reverse_tta_transform(dir_logit, rotate_degree, flip_direction)
                point_logit = self.reverse_tta_transform(point_logit, rotate_degree, flip_direction)

                sem_logit_list.append(sem_logit)
                dir_logit_list.append(dir_logit)
                point_logit_list.append(point_logit)

        sem_logit = sum(sem_logit_list) / len(sem_logit_list)
        dir_logit = sum(dir_logit_list) / len(dir_logit_list)
        point_logit = sum(point_logit_list) / len(point_logit_list)

        sem_logit = self._ddm_enhencement(sem_logit, dir_logit, point_logit)

        return sem_logit

    def split_axes(self, window_size, overlap_size, height, width):
        ws = window_size
        os = overlap_size

        i_axes = [0]
        j_axes = [0]
        cur = 0
        edge_base = ws - os // 2
        middle_base = ws - os
        while True:
            if cur == 0:
                cur += edge_base
            else:
                cur += middle_base

            i_axes.append(cur)
            j_axes.append(cur)

            if cur + edge_base == height:
                i_axes.append(cur + edge_base)
            if cur + edge_base == width:
                j_axes.append(cur + edge_base)

            if i_axes[-1] == height and j_axes[-1] == width:
                break

        return i_axes, j_axes

    def split_inference(self, img, meta, rescale):
        """using half-and-half strategy to slide inference."""
        window_size = self.test_cfg.crop_size[0]
        overlap_size = self.test_cfg.overlap_size[0]

        N, C, H, W = img.shape

        input = img

        # zero pad for border patches
        pad_h = 0
        if H - window_size > 0:
            pad_h = (window_size - overlap_size) - (H - window_size) % (window_size - overlap_size)
            tmp = torch.zeros((N, C, pad_h // 2, W), device=img.device)
            input = torch.cat((tmp, input), dim=2)
            input = torch.cat((input, tmp), dim=2)
        if W - window_size > 0:
            pad_w = (window_size - overlap_size) - (W - window_size) % (window_size - overlap_size)
            tmp = torch.zeros((N, C, H + pad_h, pad_w // 2), device=img.device)
            input = torch.cat((tmp, input), dim=3)
            input = torch.cat((input, tmp), dim=3)
        _, C1, H1, W1 = input.size()

        output = torch.zeros((input.size(0), 3, H1, W1)).to(img.device)
        output1 = torch.zeros((input.size(0), 9, H1, W1)).to(img.device)
        output2 = torch.zeros((input.size(0), 1, H1, W1)).to(img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                input_patch = input[:, :, i:r_end, j:c_end]
                input_var = input_patch / 255
                output_patch, output_patch1, output_patch2 = self.calculate(input_var)
                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]
                output1[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_patch1[:, :, ind1_s - i:ind1_e - i,
                                                                            ind2_s - j:ind2_e - j]
                output2[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_patch2[:, :, ind1_s - i:ind1_e - i,
                                                                            ind2_s - j:ind2_e - j]

        output = output[:, :, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W]
        output1 = output1[:, :, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W]
        output2 = output2[:, :, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W]
        # if rescale:
        #     output = resize(output, size=meta['ori_hw'], mode='bilinear', align_corners=False)
        # aji: 62.38 dice: 81.97
        np.save('_prob0.npy', np.array(output.cpu()))
        np.save('_prob1.npy', np.array(output1.cpu()))
        np.save('_prob2.npy', np.array(output2.cpu()))
        exit(0)
        return output, output1, output2

    # def split_inference(self, img, meta, rescale):
    #     ws = self.test_cfg.crop_size[0]
    #     os = self.test_cfg.overlap_size[0]

    #     B, C, H, W = img.shape

    #     # zero pad for border patches
    #     pad_h = 0
    #     pad_w = 0
    #     if H - ws > 0:
    #         pad_h = (ws - os) - (H - ws) % (ws - os)

    #     if W - ws > 0:
    #         pad_w = (ws - os) - (W - ws) % (ws - os)

    #     H1 = pad_h + H
    #     W1 = pad_w + W

    #     img_canvas = torch.zeros((B, C, H1, W1), dtype=img.dtype, device=img.device)
    #     img_canvas.fill_(128)
    #     img_canvas[:, :, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W] = img

    #     _, _, H1, W1 = img_canvas.shape
    #     sem_logit = torch.zeros((B, self.num_classes, H1, W1), dtype=img.dtype, device=img.device)
    #     dir_logit = torch.zeros((B, self.num_angles + 1, H1, W1), dtype=img.dtype, device=img.device)
    #     point_logit = torch.zeros((B, 1, H1, W1), dtype=img.dtype, device=img.device)

    #     i_axes, j_axes = self.split_axes(ws, os, H1, W1)

    #     for i in range(len(i_axes) - 1):
    #         for j in range(len(j_axes) - 1):
    #             r_patch_s = i_axes[i] if i == 0 else i_axes[i] - os // 2
    #             r_patch_e = r_patch_s + ws
    #             c_patch_s = j_axes[j] if j == 0 else j_axes[j] - os // 2
    #             c_patch_e = c_patch_s + ws
    #             img_patch = img_canvas[:, :, r_patch_s:r_patch_e, c_patch_s:c_patch_e]
    #             sem_patch, dir_patch, point_patch = self.calculate(img_patch)

    #             # patch overlap remove
    #             r_valid_s = i_axes[i] - r_patch_s
    #             r_valid_e = i_axes[i + 1] - r_patch_s
    #             c_valid_s = j_axes[j] - c_patch_s
    #             c_valid_e = j_axes[j + 1] - c_patch_s
    #             sem_patch = sem_patch[:, :, r_valid_s:r_valid_e, c_valid_s:c_valid_e]
    #             dir_patch = dir_patch[:, :, r_valid_s:r_valid_e, c_valid_s:c_valid_e]
    #             point_patch = point_patch[:, :, r_valid_s:r_valid_e, c_valid_s:c_valid_e]
    #             sem_logit[:, :, i_axes[i]:i_axes[i + 1], j_axes[j]:j_axes[j + 1]] = sem_patch
    #             dir_logit[:, :, i_axes[i]:i_axes[i + 1], j_axes[j]:j_axes[j + 1]] = dir_patch
    #             point_logit[:, :, i_axes[i]:i_axes[i + 1], j_axes[j]:j_axes[j + 1]] = point_patch

    #     sem_logit = sem_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
    #     dir_logit = dir_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
    #     point_logit = point_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
    #     if rescale:
    #         sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
    #         dir_logit = resize(dir_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
    #         point_logit = resize(point_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
    #     return sem_logit, dir_logit, point_logit

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        sem_logit, dir_logit, point_logit = self.calculate(img)
        if rescale:
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            dir_logit = resize(dir_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            point_logit = resize(point_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        return sem_logit, dir_logit, point_logit

    def _mask_loss(self, mask_logit, mask_gt):
        mask_loss = {}
        mask_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        mask_dice_loss_calculator = MultiClassDiceLoss(num_classes=self.num_classes)
        # Assign weight map for each pixel position
        # mask_loss *= weight_map
        mask_ce_loss = torch.mean(mask_ce_loss_calculator(mask_logit, mask_gt))
        mask_dice_loss = mask_dice_loss_calculator(mask_logit, mask_gt)
        # loss weight
        alpha = 1
        beta = 1
        mask_loss['mask_ce_loss'] = alpha * mask_ce_loss
        mask_loss['mask_dice_loss'] = beta * mask_dice_loss

        return mask_loss

    def _dir_loss(self, dir_logit, dir_gt):
        dir_loss = {}
        dir_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        dir_dice_loss_calculator = MultiClassDiceLoss(num_classes=self.num_angles + 1)
        dir_ce_loss = torch.mean(dir_ce_loss_calculator(dir_logit, dir_gt))
        dir_dice_loss = dir_dice_loss_calculator(dir_logit, dir_gt)
        # loss weight
        alpha = 1
        beta = 1
        dir_loss['dir_ce_loss'] = alpha * dir_ce_loss
        dir_loss['dir_dice_loss'] = beta * dir_dice_loss

        return dir_loss

    def _point_loss(self, point_logit, point_gt):
        point_loss = {}
        point_mse_loss_calculator = nn.MSELoss()
        point_mse_loss = point_mse_loss_calculator(point_logit, point_gt)
        # loss weight
        alpha = 1
        point_loss['point_mse_loss'] = alpha * point_mse_loss

        return point_loss

    def _training_metric(self, mask_logit, dir_logit, point_logit, mask_gt, dir_gt, point_gt):
        wrap_dict = {}
        # detach these training variable to avoid gradient noise.
        clean_mask_logit = mask_logit.clone().detach()
        clean_mask_gt = mask_gt.clone().detach()
        clean_dir_logit = dir_logit.clone().detach()
        clean_dir_gt = dir_gt.clone().detach()

        wrap_dict['mask_miou'] = miou(clean_mask_logit, clean_mask_gt, self.num_classes)
        wrap_dict['dir_miou'] = miou(clean_dir_logit, clean_dir_gt, self.num_angles + 1)

        wrap_dict['mask_tiou'] = tiou(clean_mask_logit, clean_mask_gt, self.num_classes)
        wrap_dict['dir_tiou'] = tiou(clean_dir_logit, clean_dir_gt, self.num_angles + 1)

        # metric calculate
        mask_pred = torch.argmax(mask_logit, dim=1).cpu().numpy().astype(np.uint8)
        mask_pred[mask_pred == (self.num_classes - 1)] = 0
        mask_target = mask_gt.cpu().numpy().astype(np.uint8)
        mask_target[mask_target == (self.num_classes - 1)] = 0

        N = mask_pred.shape[0]
        wrap_dict['aji'] = 0.
        for i in range(N):
            aji_single_image = aggregated_jaccard_index(mask_pred[i], mask_target[i])
            wrap_dict['aji'] += 100.0 * torch.tensor(aji_single_image)
        # distributed environment requires cuda tensor
        wrap_dict['aji'] /= N
        wrap_dict['aji'] = wrap_dict['aji'].cuda()

        return wrap_dict

    @classmethod
    def _ddm_enhencement(self, mask_logit, dir_logit, point_logit):
        # make direction differential map
        dir_map = torch.argmax(dir_logit, dim=1)
        dir_differential_map = generate_direction_differential_map(dir_map, 9)

        # using point map to remove center point direction differential map
        point_logit = point_logit[:, 0, :, :]
        point_logit = point_logit - torch.min(point_logit) / (torch.max(point_logit) - torch.min(point_logit))

        # mask out some redundant direction differential
        dir_differential_map[point_logit > 0.2] = 0

        # using direction differential map to enhance edge
        mask_logit = F.softmax(mask_logit, dim=1)
        mask_logit[:, -1, :, :] = (mask_logit[:, -1, :, :] + dir_differential_map) * (1 + 2 * dir_differential_map)

        return mask_logit