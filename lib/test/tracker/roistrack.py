import math


from lib.models.roistrack import build_roistrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class ROISTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(ROISTrack, self).__init__(params)
        network = build_roistrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu',weights_only=False)['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.lost = 0.0
        self.search_factor = self.params.search_factor
        self.refond =True

        self.last_valid_state = []
        self.last_valid_score = 0.0
        self.last_in_border = False
        self.lost_type = None  # 重置丢失类型
        self.collecting_templates=False


    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        def update(response,out_dict):
            pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
            pred_boxes = pred_boxes.view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        score_max = pred_score_map.max().item()
        lost_threshold = 0.385

        border_threshold = 0.003 * min(H, W)

        if score_max > lost_threshold and self.refond == True:
            update(response, out_dict)
            # 新增：记录最后成功跟踪的位置和状态
            self.last_valid_state = self.state.copy()
            self.last_valid_score = score_max
            self.last_in_border = self._is_near_border(self.state, H, W, border_threshold)

            self.refond = False
            self.lost = 0.0
            self.lost_type = None  # 重置丢失类型
        elif score_max > lost_threshold  and self.refond == False:
            update(response, out_dict)

            # 新增：记录最后成功跟踪的位置和状态
            self.last_valid_state = self.state.copy()
            self.last_valid_score = score_max
            self.last_in_border = self._is_near_border(self.state, H, W, border_threshold)

            self.refond = False
            self.lost = self.lost - 0.01
            self.lost_type = None  # 重置丢失类型
        else:
            if self.refond == False :
                self.search_factor = 4.0
                self.lost = 0.0
                self.refond = True
                # 新增：判断丢失类型
                if hasattr(self, 'last_in_border') and self.last_in_border:
                    self.lost_type = "out_of_view"  # 在视野边缘丢失，推测移出视野
                else:
                    self.lost_type = "in_view"  # 在视野内部丢失
                # 记录丢失时的信息
                self.lost_frame = self.frame_id
                self.lost_score = score_max
            else:
                self.lost += 1
                self.refond = True

        if self.lost <= -4.0 : #连续300不丢失;3.1875
            self.search_factor = 3.165
        elif self.lost <= -3.0 :#连续250不丢失
            self.search_factor = 3.25
        elif self.lost <= -2.0 :#连续200不丢失
            self.search_factor = 3.5
        elif self.lost <= -1.5  :self.search_factor = 3.6
        elif self.lost <= -1.0  :self.search_factor = 3.7
        elif self.lost <= -0.75 :self.search_factor = 3.8
        elif self.lost <= -0.5  :self.search_factor = 3.85
        elif self.lost <= -0.3 :self.search_factor = 3.9
        elif self.lost <= 0.0 or self.lost <10:
            self.search_factor = 4.0
            if  self.lost_type == "in_view":  # 在视野内部丢失:
                update(response, out_dict)
        elif self.lost >= 10 and self.lost <= 20 :
            self.search_factor +=0.15
            if self.lost_type == "in_view":
                update(response, out_dict)
        elif self.lost >= 20 and self.lost <= 50 :
            self.search_factor += 0.05
            update(response, out_dict)
        elif self.lost >= 50 :
            self.search_factor = 6.0
            update(response, out_dict)
        else:
            self.search_factor = self.params.search_factor
        if self.state == [0,0,W,H]:
            self.search_factor = 0.8
            self.lost =0

        self.update_threshold=1.2
        self.update_intervals=200#290

        if self.frame_id % self.update_intervals == 0 or self.frame_id <= 5:
            # 触发模板采集
            self.collecting_templates = True
            self.templates_collected = 0
            self.template_bank = []

        # 如果处于模板采集状态，则采集当前帧模板
        if self.collecting_templates:
            self.collect_current_frame_template(image, initial_score=pred_score_map.max().item(), hann_score=response.max().item())
            # 检查是否完成采集
            if self.templates_collected >= 20 or (self.templates_collected >= 5 and self.frame_id <20):
                self.select_and_update_template(initial_score_threshold=0.866, hann_score_threshold=0.851)
                self.collecting_templates = False

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(0 * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}
    def _is_near_border(self, box, H, W, threshold):
        """判断框是否靠近图像边界"""
        x, y, w, h = box
        return (x < threshold or y < threshold or
                (x+W/4 ) > (W - threshold) or (y+H/4 ) > (H - threshold))

    def collect_current_frame_template(self, image, initial_score=0.0, hann_score=0.0):
        """采集当前帧的模板"""
        # 提取模板

        z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        # 存储模板和得分
        self.template_bank.append((self.frame_id,z_patch_arr, initial_score, hann_score,z_amask_arr))
        self.templates_collected += 1
    def select_and_update_template(self,initial_score_threshold, hann_score_threshold):
        """筛选并选择最优模板"""
        # 筛选合格模板
        valid_templates = [t for t in self.template_bank if t[2] > initial_score_threshold and t[3] > hann_score_threshold]
        # 如果合格模板数量足够，选择最优模板
        if len(valid_templates) >= 5 or (self.frame_id<=20 and len(valid_templates) >= 2):
            # 按初始得分排序
            sorted_templates = sorted(valid_templates, key=lambda x: x[2], reverse=True)
            # 按Hann得分排序
            hann_sorted = sorted(valid_templates, key=lambda x: x[3], reverse=True)

            top_initial = sorted_templates[:3]
            top_hann = hann_sorted[:3]

            # 查找同时在两个列表中的模板
            candidates = []
            for t in top_initial:
                if t in top_hann:
                    # 计算综合得分（这里使用乘积，确保两者都高）
                    combined_score = t[2] * t[3]
                    candidates.append((t[0], t[1], t[2], t[3],t[4],combined_score))
            #best_template = sorted_templates[0]
            if candidates:
                best_template = sorted(candidates, key=lambda x: x[5], reverse=True)[0]
            else:
                best_template = hann_sorted[0]
            # 更新当前模板
            self.z_patch_arr = best_template[1]  # 更新模板
            z_amask_arr = best_template[4]  # 更新模板掩码
            template = self.preprocessor.process(self.z_patch_arr, z_amask_arr)
            self.z_dict1 = template



    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return ROISTrack
