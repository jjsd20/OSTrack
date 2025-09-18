"""
Basic OSTrack model.
"""
import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones


from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce

from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.timostrack import TimesNet


#from .TimesNet import TimesNetTracking


class TIMOSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head,TrackingConfig, aux_loss=False, head_type="CORNER", multi_candidate=False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.timesnet = TimesNet.TimesNetTracking(configs=TrackingConfig)
        # 新增：时序特征投影层（将TimesNet的时序特征映射到OSTrack的特征维度）
        self.temporal_proj = nn.Sequential(
            nn.Linear(TrackingConfig.d_model, TrackingConfig.d_model*2),  # 时序特征通道→视觉特征通道
            nn.ReLU(),
            nn.Unflatten(2, (TrackingConfig.d_model*2, 1, 1)),  # [B, pred_len, 128] → [B, pred_len, 128, 1, 1]
            nn.ConvTranspose2d(TrackingConfig.d_model*2, TrackingConfig.d_model*2, kernel_size=16, stride=16)  # 上采样到16×16
        )
        
        # 添加选择网络：从多个时序帧中选择最佳预测结果
        self.selection_network = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化 [B, seq, C, H, W] → [B, seq, C, 1, 1]
            nn.Flatten(start_dim=2),       # [B, seq, C, 1, 1] → [B, seq, C]
            nn.Linear(TrackingConfig.d_model*2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),              # 每个时序帧的置信度分数 [B, seq, 1]
            nn.Sigmoid()                   # 归一化到[0,1]
        )

        self.aux_loss = aux_loss
        self.head_type = head_type
        self.multi_candidate = multi_candidate
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)#16
            self.feat_len_s = int(box_head.feat_sz ** 2)#256

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                gt_sequence_anno_backward: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )
        # TimesNet处理时序信息
        temporal_output, temporal_features = self.timesnet(gt_sequence_anno_backward, None)
        # temporal_features: [B, seq, 4]
        y = self.temporal_proj[0](temporal_features)  # Linear
        y = self.temporal_proj[1](y)  # ReLU
        B, seq, C = y.shape
        y = y.view(B * seq, C, 1, 1)  # [B*seq, C, 1, 1]
        y = self.temporal_proj[3](y)  # ConvTranspose2d
        # x: [B*seq, C', H, W] 例如 [B*seq, 768, 16, 16]
        C_out, H, W = y.shape[1:]
        y = y.view(B, seq, C_out, H, W)  # [B, seq, C', H, W]

        # 使用选择网络计算每个时序帧的置信度分数
        selection_scores = self.selection_network(y)  # [B, seq, 1]
        selection_weights = torch.softmax(selection_scores.squeeze(-1), dim=1)  # [B, seq] softmax归一化
        
        # 方法1：加权融合所有时序帧
        weighted_temporal_feat = torch.sum(y * selection_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), dim=1)  # [B, C', H, W]
        
        # 方法2：选择置信度最高的单一帧（可选）
        best_frame_idx = torch.argmax(selection_weights, dim=1)  # [B]
        batch_indices = torch.arange(B, device=y.device)
        best_temporal_feat = y[batch_indices, best_frame_idx]  # [B, C', H, W]
        
        # 可以选择使用加权融合或最佳单一帧
        temporal_feat = weighted_temporal_feat  # 使用加权融合
        # temporal_feat = best_temporal_feat    # 或使用最佳单一帧
        
        # Forward head
        feat_last = x#(b,320,768)
        if isinstance(x, list):
            feat_last = x[-1]
            
        if self.multi_candidate:
            # 多候选框模式：为每个时序帧生成预测结果
            out = self.forward_head_multi_candidate(feat_last, y, selection_weights, None)
        else:
            # 单一预测模式：使用融合后的时序特征
            out = self.forward_head(feat_last, temporal_feat, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        # 添加TimesNet的预测输出用于损失计算
        out['timesnet_pred'] = temporal_output  # TimesNet预测的bbox [B, pred_len, 4]
        # 添加选择网络相关信息
        out['selection_scores'] = selection_scores  # [B, seq, 1] 每帧的原始置信度分数
        out['selection_weights'] = selection_weights  # [B, seq] 归一化后的权重
        out['best_frame_idx'] = best_frame_idx  # [B] 最佳帧索引

        return out

    def forward_head(self, cat_feature, temporal_feat,gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()#[32, 1, 768, 256]
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)#[32, 768, 16, 16]
        fused_feat = torch.cat([opt_feat, temporal_feat], dim=1)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(fused_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError
    
    def forward_head_multi_candidate(self, cat_feature, temporal_feats, selection_weights, gt_score_map=None):
        """
        多候选框预测模式
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        temporal_feats: [B, seq, C', H, W] 所有时序帧特征
        selection_weights: [B, seq] 每帧的权重
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()  # [B, 1, C, HW]
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)  # [B, C, H, W]
        
        B, seq, C_temp, H_temp, W_temp = temporal_feats.shape
        
        # 为每个时序帧生成预测结果
        all_pred_boxes = []
        all_score_maps = []
        
        for t in range(seq):
            temp_feat = temporal_feats[:, t, :, :, :]  # [B, C', H, W]
            fused_feat = torch.cat([opt_feat, temp_feat], dim=1)  # [B, C+C', H, W]
            
            if self.head_type == "CORNER":
                pred_box, score_map = self.box_head(opt_feat, True)
                outputs_coord = box_xyxy_to_cxcywh(pred_box)
                outputs_coord_new = outputs_coord.view(bs, Nq, 4)
                all_pred_boxes.append(outputs_coord_new)
                all_score_maps.append(score_map)
                
            elif self.head_type == "CENTER":
                score_map_ctr, bbox, size_map, offset_map = self.box_head(fused_feat, gt_score_map)
                outputs_coord = bbox
                outputs_coord_new = outputs_coord.view(bs, Nq, 4)
                all_pred_boxes.append(outputs_coord_new)
                all_score_maps.append(score_map_ctr)
        
        # 堆叠所有候选框 [B, seq, Nq, 4]
        all_pred_boxes = torch.stack(all_pred_boxes, dim=1)  # [B, seq, Nq, 4]
        all_score_maps = torch.stack(all_score_maps, dim=1)  # [B, seq, ...]
        
        # 使用选择权重计算最终预测结果
        # 方法1：加权平均
        final_pred_boxes = torch.sum(all_pred_boxes * selection_weights.unsqueeze(-1).unsqueeze(-1), dim=1)  # [B, Nq, 4]
        
        # 方法2：选择最佳帧的预测结果
        best_frame_idx = torch.argmax(selection_weights, dim=1)  # [B]
        batch_indices = torch.arange(B, device=temporal_feats.device)
        best_pred_boxes = all_pred_boxes[batch_indices, best_frame_idx]  # [B, Nq, 4]
        best_score_maps = all_score_maps[batch_indices, best_frame_idx]  # [B, ...]
        
        if self.head_type == "CORNER":
            out = {
                'pred_boxes': final_pred_boxes,  # 或使用 best_pred_boxes
                'score_map': best_score_maps,
                'all_pred_boxes': all_pred_boxes,  # 所有候选框
                'all_score_maps': all_score_maps,  # 所有分数图
            }
        elif self.head_type == "CENTER":
            out = {
                'pred_boxes': final_pred_boxes,  # 或使用 best_pred_boxes
                'score_map': best_score_maps,
                'all_pred_boxes': all_pred_boxes,  # 所有候选框
                'all_score_maps': all_score_maps,  # 所有分数图
            }
        
        return out


def build_timostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('TIMOSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, #0.1
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC, #[3,6,9]
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,#[0.7,0.7,0.7]
                                           )
        hidden_dim = backbone.embed_dim + cfg.TIMING.d_model*2  # 768 + 128 = 896
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = TIMOSTrack(
        backbone,
        box_head,
        TrackingConfig=cfg.TIMING,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        multi_candidate=cfg.MODEL.get('MULTI_CANDIDATE', False),  # 从配置中读取，默认为False
    )

    if 'TIMOSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
