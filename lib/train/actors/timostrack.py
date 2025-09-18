from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class TIMOSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        gt_sequence_anno_backward=data['gt_sequence_anno_backward'].view(-1, *data['gt_sequence_anno_backward'].shape[2:])#历史帧序列的标签
        #gt_sequence_anno_forward=data['gt_sequence_anno_forward']

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])#[32,64]

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH#20
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH#80
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            gt_sequence_anno_backward=gt_sequence_anno_backward,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1] #[32,4] # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
        gt_sequence_anno_forward=gt_dict['gt_sequence_anno_forward'].squeeze(0)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']#[32,1,4]
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        
        # compute selection network loss (替代原来的TimesNet损失)
        selection_loss = torch.tensor(0.0, device=l1_loss.device)
        
        # 计算选择网络损失
        if 'selection_weights' in pred_dict and 'timesnet_pred' in pred_dict:
            selection_loss = self.compute_selection_loss(pred_dict, gt_dict)
        
        # 计算多候选框损失（如果启用）
        multi_candidate_loss = torch.tensor(0.0, device=l1_loss.device)
        if 'all_pred_boxes' in pred_dict:
            multi_candidate_loss = self.compute_multi_candidate_loss(pred_dict, gt_dict)
        
        # weighted sum (移除冗余的TimesNet损失)
        loss = (self.loss_weight['giou'] * giou_loss + 
                self.loss_weight['l1'] * l1_loss + 
                self.loss_weight['focal'] * location_loss +
                self.loss_weight.get('selection', 0.1) * selection_loss +
                self.loss_weight.get('multi_candidate', 0.05) * multi_candidate_loss)
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/selection": selection_loss.item(),
                      "Loss/multi_candidate": multi_candidate_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
    
    def compute_selection_loss(self, pred_dict, gt_dict):
        """
        计算选择网络的损失
        选择网络应该选择与当前GT最接近的时序帧
        """
        # 获取选择权重 [B, seq]
        selection_weights = pred_dict['selection_weights']  
        
        # 获取TimesNet预测的所有帧 [B, pred_len, 4]
        timesnet_pred = pred_dict['timesnet_pred']
        
        # 获取当前帧的GT
        gt_bbox = gt_dict['search_anno'][-1]  # [B, 4] (x1,y1,w,h)
        gt_bbox_xyxy = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0, max=1.0)
        
        B, seq_len, _ = timesnet_pred.shape
        
        # 计算每个时序帧预测与GT的IoU作为监督信号
        iou_scores = []
        
        for t in range(seq_len):
            # 当前时序帧的预测 [B, 4]
            frame_pred = timesnet_pred[:, t, :]  # (x, y, w, h)
            frame_pred_xyxy = box_xywh_to_xyxy(frame_pred).clamp(min=0.0, max=1.0)
            
            # 计算IoU
            try:
                from lib.utils.box_ops import box_iou
                iou = torch.diag(box_iou(frame_pred_xyxy, gt_bbox_xyxy)[0])  # [B]
                iou_scores.append(iou)
            except:
                # 如果没有box_iou函数，使用简单的L1距离作为替代
                l1_dist = torch.mean(torch.abs(frame_pred_xyxy - gt_bbox_xyxy), dim=1)  # [B]
                # 将距离转换为相似度分数（距离越小，分数越高）
                similarity = torch.exp(-l1_dist * 5.0)  # [B]
                iou_scores.append(similarity)
        
        # 堆叠所有IoU分数 [B, seq]
        iou_scores = torch.stack(iou_scores, dim=1)  # [B, seq]
        
        # 使用softmax归一化IoU分数作为目标权重
        target_weights = torch.softmax(iou_scores, dim=1)  # [B, seq]
        
        # 计算选择权重与目标权重之间的KL散度损失
        # 为了数值稳定性，添加小的epsilon
        eps = 1e-8
        selection_weights_safe = selection_weights + eps
        target_weights_safe = target_weights + eps
        
        # KL散度损失：KL(target || selection)
        kl_loss = torch.sum(target_weights_safe * torch.log(target_weights_safe / selection_weights_safe), dim=1)
        
        # 也可以使用L2损失作为替代
        l2_loss = torch.mean((selection_weights - target_weights) ** 2, dim=1)
        
        # 结合两种损失
        selection_loss = torch.mean(0.7 * kl_loss + 0.3 * l2_loss)
        
        return selection_loss
    
    def compute_multi_candidate_loss(self, pred_dict, gt_dict):
        """
        多候选框模式的损失计算
        为每个候选框计算损失，然后加权平均
        """
        if 'all_pred_boxes' not in pred_dict:
            return torch.tensor(0.0, device=pred_dict['pred_boxes'].device)
        
        all_pred_boxes = pred_dict['all_pred_boxes']  # [B, seq, Nq, 4]
        selection_weights = pred_dict['selection_weights']  # [B, seq]
        
        # 获取GT
        gt_bbox = gt_dict['search_anno'][-1]  # [B, 4]
        gt_bbox_xyxy = box_xywh_to_xyxy(gt_bbox).clamp(min=0.0, max=1.0)
        
        B, seq, Nq, _ = all_pred_boxes.shape
        
        # 为每个候选框计算损失
        candidate_losses = []
        
        for t in range(seq):
            # 当前时序帧的预测框 [B, Nq, 4]
            frame_pred_boxes = all_pred_boxes[:, t, :, :]  # [B, Nq, 4] (cx, cy, w, h)
            frame_pred_xyxy = box_cxcywh_to_xyxy(frame_pred_boxes).view(-1, 4)  # [B*Nq, 4]
            
            # 扩展GT到匹配维度
            gt_boxes_expanded = gt_bbox_xyxy[:, None, :].repeat(1, Nq, 1).view(-1, 4)  # [B*Nq, 4]
            
            # 计算L1损失
            l1_loss = self.objective['l1'](frame_pred_xyxy, gt_boxes_expanded)
            
            # 计算GIoU损失
            try:
                giou_loss, _ = self.objective['giou'](frame_pred_xyxy, gt_boxes_expanded)
            except:
                giou_loss = torch.tensor(0.0, device=l1_loss.device)
            
            # 综合损失
            frame_loss = self.loss_weight['l1'] * l1_loss + self.loss_weight['giou'] * giou_loss
            candidate_losses.append(frame_loss)
        
        # 堆叠所有候选框损失 [seq]
        candidate_losses = torch.stack(candidate_losses)  # [seq]
        
        # 使用选择权重加权平均
        weighted_loss = torch.sum(candidate_losses * selection_weights.mean(dim=0))  # 平均批次维度
        
        return weighted_loss
