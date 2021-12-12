import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tiseg.utils import resize
from tiseg.utils.evaluation.metrics import aggregated_jaccard_index
from ..backbones import TorchVGG16BN
from ..builder import SEGMENTORS
from ..heads.cd_head import CDHead
from ..losses import miou, tiou, MultiClassDiceLoss
from ..utils import generate_direction_differential_map
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class CDNetSegmentor(BaseSegmentor):
    """Base class for segmentors."""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(CDNetSegmentor, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.num_angles = 8

        self.backbone = TorchVGG16BN(in_channels=3, pretrained=True, out_indices=[0, 1, 2, 3, 4])
        self.head = CDHead(
            num_classes=self.num_classes,
            num_angles=self.num_angles,
            in_dims=(64, 128, 256, 512, 512),
            stage_dims=[16, 32, 64, 128, 256],
            dropout_rate=0.1,
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'),
            align_corners=False)

    def calculate(self, img):
        img_feats = self.backbone(img)
        mask_logit, dir_logit, point_logit = self.head(img_feats)

        if self.test_cfg.get('use_ddm', False):
            # The whole image is too huge. So we use slide inference in
            # default.
            mask_logit = resize(input=mask_logit, size=self.test_cfg['plane_size'])
            direction_logit = resize(input=dir_logit, size=self.test_cfg['plane_size'])
            point_logit = resize(input=point_logit, size=self.test_cfg['plane_size'])

            mask_logit = self._ddm_enhencement(mask_logit, direction_logit, point_logit)
        mask_logit = resize(input=mask_logit, size=img.shape[2:], mode='bilinear', align_corners=False)

        return mask_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if self.training:
            img_feats = self.backbone(data['img'])
            mask_logit, dir_logit, point_logit = self.head(img_feats)

            assert label is not None
            mask_gt = label['sem_gt']
            point_gt = label['point_gt']
            dir_gt = label['dir_gt']

            loss = dict()
            mask_logit = resize(input=mask_logit, size=mask_gt.shape[2:])
            direction_logit = resize(input=dir_logit, size=dir_gt.shape[2:])
            point_logit = resize(input=point_logit, size=point_gt.shape[2:])

            mask_gt = mask_gt.squeeze(1)
            dir_gt = dir_gt.squeeze(1)

            # TODO: Conside to remove some edge loss value.
            # mask branch loss calculation
            mask_loss = self._mask_loss(mask_logit, mask_gt)
            loss.update(mask_loss)
            # direction branch loss calculation
            direction_loss = self._direction_loss(direction_logit, dir_gt)
            loss.update(direction_loss)
            # point branch loss calculation
            point_loss = self._point_loss(point_logit, point_gt)
            loss.update(point_loss)

            # calculate training metric
            training_metric_dict = self._training_metric(mask_logit, direction_logit, point_logit, mask_gt, dir_gt,
                                                         point_gt)
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
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            meta (dict): Image info dict where each dict has: 'img_info',
                'ann_info'
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        raw_img = img
        assert self.test_cfg.mode in ['slide', 'whole']

        self.rotate_degrees = self.test_cfg.get('rotate_degrees', [0])
        self.flip_directions = self.test_cfg.get('flip_directions', ['none'])
        seg_logit_list = []

        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                rotate_num = (rotate_degree // 90) % 4
                img = torch.rot90(raw_img, k=rotate_num, dims=(-2, -1))

                if flip_direction == 'horizontal':
                    img = torch.flip(img, dims=[-1])
                if flip_direction == 'vertical':
                    img = torch.flip(img, dims=[-2])
                if flip_direction == 'diagonal':
                    img = torch.flip(img, dims=[-2, -1])

                if self.test_cfg.mode == 'slide':
                    seg_logit = self.split_inference(img, meta, rescale)
                else:
                    seg_logit = self.whole_inference(img, meta, rescale)

                if flip_direction == 'horizontal':
                    seg_logit = torch.flip(seg_logit, dims=[-1])
                if flip_direction == 'vertical':
                    seg_logit = torch.flip(seg_logit, dims=[-2])
                if flip_direction == 'diagonal':
                    seg_logit = torch.flip(seg_logit, dims=[-2, -1])

                rotate_num = 4 - rotate_num
                seg_logit = torch.rot90(seg_logit, k=rotate_num, dims=(-2, -1))

                seg_logit_list.append(seg_logit)

        seg_logit = sum(seg_logit_list) / len(seg_logit_list)

        return seg_logit

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
        ws = self.test_cfg.crop_size[0]
        os = (self.test_cfg.crop_size[0] - self.test_cfg.stride[0])

        B, C, H, W = img.shape

        # zero pad for border patches
        pad_h = 0
        pad_w = 0
        if H - ws > 0:
            pad_h = (ws - os) - (H - ws) % (ws - os)

        if W - ws > 0:
            pad_w = (ws - os) - (W - ws) % (ws - os)

        H1 = pad_h + H
        W1 = pad_w + W

        img_canvas = torch.zeros((B, C, H1, W1), dtype=img.dtype, device=img.device)
        img_canvas.fill_(128)
        img_canvas[:, :, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W] = img

        _, _, H1, W1 = img_canvas.shape
        sem_output = torch.zeros((B, self.num_classes, H1, W1))

        i_axes, j_axes = self.split_axes(ws, os, H1, W1)

        for i in range(len(i_axes) - 1):
            for j in range(len(j_axes) - 1):
                r_patch_s = i_axes[i] if i == 0 else i_axes[i] - os // 2
                r_patch_e = r_patch_s + ws
                c_patch_s = j_axes[j] if j == 0 else j_axes[j] - os // 2
                c_patch_e = c_patch_s + ws
                img_patch = img_canvas[:, :, r_patch_s:r_patch_e, c_patch_s:c_patch_e]
                sem_patch = self.calculate(img_patch)

                # patch overlap remove
                r_valid_s = i_axes[i] - r_patch_s
                r_valid_e = i_axes[i + 1] - r_patch_s
                c_valid_s = j_axes[j] - c_patch_s
                c_valid_e = j_axes[j + 1] - c_patch_s
                sem_patch = sem_patch[:, :, r_valid_s:r_valid_e, c_valid_s:c_valid_e]
                sem_output[:, :, i_axes[i]:i_axes[i + 1], j_axes[j]:j_axes[j + 1]] = sem_patch

        sem_output = sem_output[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        if rescale:
            sem_output = resize(sem_output, size=meta['ori_hw'], mode='bilinear', align_corners=False)
        return sem_output

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        seg_logit = self.calculate(img)
        if rescale:
            seg_logit = resize(seg_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        return seg_logit

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

    def _direction_loss(self, direction_logit, dir_gt):
        direction_loss = {}
        direction_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        direction_dice_loss_calculator = MultiClassDiceLoss(num_classes=self.num_angles + 1)
        direction_ce_loss = torch.mean(direction_ce_loss_calculator(direction_logit, dir_gt))
        direction_dice_loss = direction_dice_loss_calculator(direction_logit, dir_gt)
        # loss weight
        alpha = 1
        beta = 1
        direction_loss['direction_ce_loss'] = alpha * direction_ce_loss
        direction_loss['direction_dice_loss'] = beta * direction_dice_loss

        return direction_loss

    def _point_loss(self, point_logit, point_gt):
        point_loss = {}
        point_mse_loss_calculator = nn.MSELoss()
        point_mse_loss = point_mse_loss_calculator(point_logit, point_gt)
        # loss weight
        alpha = 1
        point_loss['point_mse_loss'] = alpha * point_mse_loss

        return point_loss

    def _training_metric(self, mask_logit, direction_logit, point_logit, mask_gt, dir_gt, point_gt):
        wrap_dict = {}
        # detach these training variable to avoid gradient noise.
        clean_mask_logit = mask_logit.clone().detach()
        clean_mask_gt = mask_gt.clone().detach()
        clean_direction_logit = direction_logit.clone().detach()
        clean_dir_gt = dir_gt.clone().detach()

        wrap_dict['mask_miou'] = miou(clean_mask_logit, clean_mask_gt, self.num_classes)
        wrap_dict['direction_miou'] = miou(clean_direction_logit, clean_dir_gt, self.num_angles + 1)

        wrap_dict['mask_tiou'] = tiou(clean_mask_logit, clean_mask_gt, self.num_classes)
        wrap_dict['direction_tiou'] = tiou(clean_direction_logit, clean_dir_gt, self.num_angles + 1)

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
    def _ddm_enhencement(self, mask_logit, direction_logit, point_logit):
        # make direction differential map
        direction_map = torch.argmax(direction_logit, dim=1)
        direction_differential_map = generate_direction_differential_map(direction_map, 9)

        # using point map to remove center point direction differential map
        point_logit = point_logit[:, 0, :, :]
        point_logit = point_logit - torch.min(point_logit) / (torch.max(point_logit) - torch.min(point_logit))

        # mask out some redundant direction differential
        direction_differential_map[point_logit > 0.2] = 0

        # using direction differential map to enhance edge
        mask_logit = F.softmax(mask_logit, dim=1)
        mask_logit[:, -1, :, :] = (mask_logit[:, -1, :, :] +
                                   direction_differential_map) * (1 + 2 * direction_differential_map)

        return mask_logit
