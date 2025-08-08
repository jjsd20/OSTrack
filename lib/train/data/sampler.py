import random
import torch.utils.data
from lib.utils import TensorDict
import numpy as np
import os
import json
from datetime import datetime


def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. 

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch#60000
        self.max_gap = max_gap #200
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        if self.train_cls:
            return self.getitem_cls()
        else:
            return self.getitem()

    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames)
                        # Increase gap until a frame is found
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros((H, W))] * self.num_template_frames
                search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros((H, W))] * self.num_search_frames

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})
                # make data augmentation
                data = self.processing(data)

                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def getitem_cls(self):
        # get data for classification
        """
        args:
            index (int): Index (Ignored since we sample randomly)
            aux (bool): whether the current data is for auxiliary use (e.g. copy-and-paste)

        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        label = None
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
            # sample template and search frame ids
            if is_video_dataset:
                if self.frame_sample_mode in ["trident", "trident_pro"]:
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                # "try" is used to handle trackingnet data failure
                # get images and bounding boxes (for templates)
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids,
                                                                                    seq_info_dict)
                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros(
                    (H, W))] * self.num_template_frames
                # get images and bounding boxes (for searches)
                # positive samples
                if random.random() < self.pos_prob:
                    label = torch.ones(1,)
                    search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames
                # negative samples
                else:
                    label = torch.zeros(1,)
                    if is_video_dataset:
                        search_frame_ids = self._sample_visible_ids(visible, num_ids=1, force_invisible=True)
                        if search_frame_ids is None:
                            search_frames, search_anno, meta_obj_test = self.get_one_search()
                        else:
                            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids,
                                                                                           seq_info_dict)
                            search_anno["bbox"] = [self.get_center_box(H, W)]
                    else:
                        search_frames, search_anno, meta_obj_test = self.get_one_search()
                    H, W, _ = search_frames[0].shape
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames

                # Sample GT sequence for TimesNet training
                gt_sequence_anno_backward, _ = self.sample_gt_sequence(
                    dataset, seq_id, template_frame_ids, search_frame_ids, seq_info_dict, min_interval=50, direction='backward', future_steps=50)
                gt_sequence_anno_forward, _ = self.sample_gt_sequence(
                    dataset, seq_id, template_frame_ids, search_frame_ids, seq_info_dict, min_interval=50, direction='forward', future_steps=30)

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   #'gt_sequence_images': gt_sequence_frames,
                                   'gt_sequence_anno_backward': gt_sequence_anno_backward,
                                   'gt_sequence_anno_forward': gt_sequence_anno_forward,
                                   #'gt_sequence_masks': gt_sequence_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})

                # make data augmentation
                data = self.processing(data)
                # add classification label
                data["label"] = label
                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def get_center_box(self, H, W, ratio=1/8):
        cx, cy, w, h = W/2, H/2, W * ratio, H * ratio
        return torch.tensor([int(cx-w/2), int(cy-h/2), int(w), int(h)])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict

    def get_one_search(self):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()
        # sample a sequence
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
        # sample a frame
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_frame_ids = self._sample_visible_ids(seq_info_dict["valid"], num_ids=1)
            else:
                search_frame_ids = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
        else:
            search_frame_ids = [1]
        # get the image, bounding box and other info
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        return search_frames, search_anno, meta_obj_test

    def get_frame_ids_trident(self, visible):
        # get template and search ids in a 'trident' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id,
                                                    allow_invisible=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        # get template and search ids in a 'stark' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids


class TimingTrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches with GT sequence support.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    Additionally, this sampler also samples a GT sequence from template start to search end with minimum interval
    for TimesNet training.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5,min_interval=50,future_steps=1):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch  # 60000
        self.max_gap = max_gap  # 200
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode
        self.min_interval=min_interval
        self.future_steps=future_steps

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        if self.train_cls:
            return self.getitem_cls()
        else:
            return self.getitem()

    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1,
                                                                 min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                    max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                    num_ids=self.num_search_frames)
                        # Increase gap until a frame is found
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                # Optimized: collect all frame IDs and get them in one call
                (template_frames, template_anno, meta_obj_train,
                 search_frames, search_anno, meta_obj_test,
                 gt_sequence_anno_backward, gt_frame_ids_backward,
                 gt_sequence_anno_forward, gt_frame_ids_forward) = self.get_all_frames_optimized(
                    dataset, seq_id, template_frame_ids, search_frame_ids, seq_info_dict,
                    min_interval=self.min_interval, future_steps=self.future_steps)

                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros(
                    (H, W))] * self.num_template_frames
                search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                    (H, W))] * self.num_search_frames

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   #'gt_sequence_images': gt_sequence_frames,
                                   'gt_sequence_anno_backward': gt_sequence_anno_backward,
                                   'gt_sequence_anno_forward': gt_sequence_anno_forward,
                                   #'gt_sequence_masks': gt_sequence_masks,
                                   # 'template_frame_ids': template_frame_ids,
                                   # 'search_frame_ids': search_frame_ids,
                                   # 'gt_frame_ids_backward': gt_frame_ids_backward,
                                   # 'gt_frame_ids_forward': gt_frame_ids_forward,
                                   # 'dataset': dataset.get_name(),
                                   # 'test_class': meta_obj_test.get('object_class_name'),
                                   'is_video_dataset': is_video_dataset,
                                   'image_size': [H,W]})
                # make data augmentation
                data = self.processing(data)

                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def getitem_cls(self):
        # get data for classification
        """
        args:
            index (int): Index (Ignored since we sample randomly)
            aux (bool): whether the current data is for auxiliary use (e.g. copy-and-paste)

        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        label = None
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
            # sample template and search frame ids
            if is_video_dataset:
                if self.frame_sample_mode in ["trident", "trident_pro"]:
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                # "try" is used to handle trackingnet data failure
                # get images and bounding boxes (for templates)
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids,
                                                                                    seq_info_dict)
                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros(
                    (H, W))] * self.num_template_frames
                # get images and bounding boxes (for searches)
                # positive samples
                if random.random() < self.pos_prob:
                    label = torch.ones(1, )
                    search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids,
                                                                                   seq_info_dict)
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames
                # negative samples
                else:
                    label = torch.zeros(1, )
                    if is_video_dataset:
                        search_frame_ids = self._sample_visible_ids(visible, num_ids=1, force_invisible=True)
                        if search_frame_ids is None:
                            search_frames, search_anno, meta_obj_test = self.get_one_search()
                        else:
                            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids,
                                                                                           seq_info_dict)
                            search_anno["bbox"] = [self.get_center_box(H, W)]
                    else:
                        search_frames, search_anno, meta_obj_test = self.get_one_search()
                    H, W, _ = search_frames[0].shape
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames

                # Sample GT sequence for TimesNet training
                gt_sequence_anno_backward, _ = self.sample_gt_sequence(
                    dataset, seq_id, template_frame_ids, search_frame_ids, seq_info_dict, min_interval=50,
                    direction='backward', future_steps=50)

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   #'gt_sequence_images': gt_sequence_frames,
                                   'gt_sequence_anno': gt_sequence_anno_backward,
                                   #'gt_sequence_masks': gt_sequence_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})

                # make data augmentation
                data = self.processing(data)
                # add classification label
                data["label"] = label
                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def get_center_box(self, H, W, ratio=1 / 8):
        cx, cy, w, h = W / 2, H / 2, W * ratio, H * ratio
        return torch.tensor([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict

    def get_one_search(self):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()
        # sample a sequence
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
        # sample a frame
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_frame_ids = self._sample_visible_ids(seq_info_dict["valid"], num_ids=1)
            else:
                search_frame_ids = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
        else:
            search_frame_ids = [1]
        # get the image, bounding box and other info
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        return search_frames, search_anno, meta_obj_test

    def get_frame_ids_trident(self, visible):
        # get template and search ids in a 'trident' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id,
                                                    allow_invisible=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        # get template and search ids in a 'stark' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def sample_gt_sequence(self, dataset, seq_id, template_frame_ids, search_frame_ids, seq_info_dict, min_interval=50, direction='backward', future_steps=5):
        """
        Sample GT sequence for TimesNet training
        direction: 'backward' - from template start to search end (历史序列)
                   'forward' - from search end to future (预测序列)
        min_interval: minimum interval for backward sequence
        future_steps: number of future steps to predict
        """
        is_video_dataset = dataset.is_video_sequence()
        if not is_video_dataset:
            # For image dataset,只返回bbox序列，补足指定数量
            template_frames, template_anno, _ = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
            bboxes = template_anno['bbox']
            target_length = min_interval if direction == 'backward' else future_steps
            bboxes = bboxes * (target_length // len(bboxes)) + bboxes[:target_length % len(bboxes)]
            # 返回张量而不是列表，确保维度正确
            return torch.stack(bboxes[:target_length]).unsqueeze(0), []

        # For video dataset, sample GT sequence
        template_start = min(template_frame_ids)
        search_end = max(search_frame_ids)
        seq_info = dataset.get_sequence_info(seq_id)
        total_frames = seq_info["length"] if "length" in seq_info else len(seq_info["visible"])
        
        if direction == 'backward':
            # 历史序列：从模板开始到搜索帧前一帧
            gt_frame_ids = list(range(template_start, search_end))
            if len(gt_frame_ids) >= min_interval:
                # 如果实际间距大于最小间距，则均匀采样
                gt_frame_ids = np.linspace(template_start, search_end - 1, min_interval, dtype=int).tolist()
            # 向前扩展
            if len(gt_frame_ids) < min_interval:
                need = min_interval - len(gt_frame_ids)
                new_start = max(0, template_start - need)
                gt_frame_ids = list(range(new_start, search_end))
            # 还不足指定帧数，说明整个视频都不够
            if len(gt_frame_ids) < min_interval:
                # 获取已有的GT序列数据
                gt_sequence_frames, gt_sequence_anno, _ = dataset.get_frames(seq_id, gt_frame_ids, seq_info_dict)
                gt_bboxes = gt_sequence_anno['bbox']
                
                # 用最前面的帧来复制填充（backward方向）
                if len(gt_bboxes) > 0:
                    first_bbox = gt_bboxes[0]
                    first_frame_id = gt_frame_ids[0]
                    # 用第一个bbox和frame_id填充
                    bboxes =  [first_bbox] * (min_interval - len(gt_bboxes))+gt_bboxes
                    complete_frame_ids =  [first_frame_id] * (min_interval - len(gt_frame_ids))+gt_frame_ids
                else:
                    # 如果连一个帧都没有，用模板帧填充
                    template_frames, template_anno, _ = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                    template_bbox = template_anno['bbox'][0]
                    bboxes = [template_bbox] * min_interval
                    complete_frame_ids = [template_frame_ids[0]] * min_interval
                
                # 返回张量而不是列表，确保维度正确
                return torch.stack(bboxes[:min_interval]).unsqueeze(0), complete_frame_ids
        else:
            # 预测序列：从搜索帧开始到未来
            gt_frame_ids = list(range(search_end, min(search_end + future_steps, total_frames)))
            # 如果后面帧数不足指定步数，向后扩展（复制最后一个未来帧）
            if len(gt_frame_ids) < future_steps:
                # 先获取已有的未来帧
                if len(gt_frame_ids) > 0:
                    gt_sequence_frames, gt_sequence_anno, _ = dataset.get_frames(seq_id, gt_frame_ids, seq_info_dict)
                    gt_bboxes = gt_sequence_anno['bbox']
                    # 用最后一个未来帧的bbox作为填充
                    last_future_bbox = gt_bboxes[-1]
                else:
                    gt_bboxes = []
                    # 如果没有未来帧，则用搜索帧的bbox作为填充
                    search_frames, search_anno, _ = dataset.get_frames(seq_id, [search_end], seq_info_dict)
                    last_future_bbox = search_anno['bbox'][0]
                
                # 补足到指定步数
                bboxes = gt_bboxes + [last_future_bbox] * (future_steps - len(gt_bboxes))
                # 生成完整的帧ID列表，确保有足够的帧ID
                complete_frame_ids = gt_frame_ids + [gt_frame_ids[-1] if gt_frame_ids else search_end] * (future_steps - len(gt_frame_ids))
                # 返回张量而不是列表，确保维度正确
                return torch.stack(bboxes[:future_steps]).unsqueeze(0), complete_frame_ids
        
        # 正常返回            
        gt_sequence_frames, gt_sequence_anno, _ = dataset.get_frames(seq_id, gt_frame_ids, seq_info_dict)
        gt_bboxes = gt_sequence_anno['bbox']
        target_length = min_interval if direction == 'backward' else future_steps
        
        # 确保bboxes和frame_ids长度一致
        if len(gt_bboxes) < target_length:
            if direction == 'backward':
                # backward序列用最前面的帧填充
                gt_bboxes = [gt_bboxes[0]] * (target_length - len(gt_bboxes))+gt_bboxes
            else:
                # forward序列用最后一个帧填充
                gt_bboxes = gt_bboxes + [gt_bboxes[-1]] * (target_length - len(gt_bboxes))
        if len(gt_frame_ids) < target_length:
            if direction == 'backward':
                # backward序列用最前面的帧ID填充
                gt_frame_ids =  [gt_frame_ids[0]] * (target_length - len(gt_frame_ids)) + gt_frame_ids
            else:
                # forward序列用最后一个帧ID填充
                gt_frame_ids = gt_frame_ids + [gt_frame_ids[-1]] * (target_length - len(gt_frame_ids))
        
        # 返回张量而不是列表，确保维度正确
        return torch.stack(gt_bboxes[:target_length]).unsqueeze(0), gt_frame_ids[:target_length]

    def sample_gt_sequence_bidirectional(self, dataset, seq_id, template_frame_ids, search_frame_ids, seq_info_dict, min_interval=50, future_steps=5):
        """
        Optimized version that samples both backward and forward GT sequences in one call
        to reduce data loading time by avoiding duplicate operations.
        
        Returns:
            tuple: (gt_sequence_anno_backward, gt_frame_ids_backward, 
                   gt_sequence_anno_forward, gt_frame_ids_forward)
        """
        is_video_dataset = dataset.is_video_sequence()
        
        if not is_video_dataset:
            # For image dataset, handle both directions
            template_frames, template_anno, _ = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
            bboxes = template_anno['bbox']
            
            # Backward sequence
            bboxes_backward = bboxes * (min_interval // len(bboxes)) + bboxes[:min_interval % len(bboxes)]
            backward_result = torch.stack(bboxes_backward[:min_interval]).unsqueeze(0)
            
            # Forward sequence  
            bboxes_forward = bboxes * (future_steps // len(bboxes)) + bboxes[:future_steps % len(bboxes)]
            forward_result = torch.stack(bboxes_forward[:future_steps]).unsqueeze(0)
            
            return backward_result, [], forward_result, []

        # For video dataset - get common info once
        template_start = min(template_frame_ids)
        search_end = max(search_frame_ids)
        seq_info = dataset.get_sequence_info(seq_id)
        total_frames = seq_info["length"] if "length" in seq_info else len(seq_info["visible"])
        
        # Sample backward sequence (历史序列：从模板开始到搜索帧前一帧)
        gt_frame_ids_backward = list(range(template_start, search_end))
        if len(gt_frame_ids_backward) >= min_interval:
            gt_frame_ids_backward = np.linspace(template_start, search_end - 1, min_interval, dtype=int).tolist()
        
        # Extend backward if needed
        if len(gt_frame_ids_backward) < min_interval:
            need = min_interval - len(gt_frame_ids_backward)
            new_start = max(0, template_start - need)
            gt_frame_ids_backward = list(range(new_start, search_end))
        
        # Sample forward sequence (预测序列：从搜索帧开始到未来)
        gt_frame_ids_forward = list(range(search_end, min(search_end + future_steps, total_frames)))
        
        # Collect all unique frame IDs to minimize dataset.get_frames calls
        all_frame_ids = set(gt_frame_ids_backward + gt_frame_ids_forward)
        
        # Handle insufficient frames for backward
        if len(gt_frame_ids_backward) < min_interval:
            # Get available frames
            available_backward_ids = [fid for fid in gt_frame_ids_backward if fid in all_frame_ids]
            if available_backward_ids:
                all_frame_ids.update(available_backward_ids)
            else:
                # Fallback to template frames
                all_frame_ids.update(template_frame_ids)
        
        # Handle insufficient frames for forward  
        if len(gt_frame_ids_forward) < future_steps:
            if gt_frame_ids_forward:
                all_frame_ids.update(gt_frame_ids_forward)
            else:
                # Fallback to search frames
                all_frame_ids.add(search_end)
        
        # Single call to get all needed frames
        all_frame_ids_list = sorted(list(all_frame_ids))
        if all_frame_ids_list:
            all_frames, all_anno, _ = dataset.get_frames(seq_id, all_frame_ids_list, seq_info_dict)
            frame_id_to_bbox = {fid: bbox for fid, bbox in zip(all_frame_ids_list, all_anno['bbox'])}
        else:
            frame_id_to_bbox = {}
        
        # Process backward sequence
        if len(gt_frame_ids_backward) < min_interval:
            if gt_frame_ids_backward and gt_frame_ids_backward[0] in frame_id_to_bbox:
                first_bbox = frame_id_to_bbox[gt_frame_ids_backward[0]]
                first_frame_id = gt_frame_ids_backward[0]
                bboxes_backward = [first_bbox] * (min_interval - len(gt_frame_ids_backward))
                bboxes_backward.extend([frame_id_to_bbox[fid] for fid in gt_frame_ids_backward if fid in frame_id_to_bbox])
                complete_frame_ids_backward = [first_frame_id] * (min_interval - len(gt_frame_ids_backward)) + gt_frame_ids_backward
            else:
                # Fallback to template
                template_frames, template_anno, _ = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                template_bbox = template_anno['bbox'][0]
                bboxes_backward = [template_bbox] * min_interval
                complete_frame_ids_backward = [template_frame_ids[0]] * min_interval
            
            backward_result = torch.stack(bboxes_backward[:min_interval]).unsqueeze(0)
            gt_frame_ids_backward = complete_frame_ids_backward
        else:
            bboxes_backward = [frame_id_to_bbox[fid] for fid in gt_frame_ids_backward if fid in frame_id_to_bbox]
            if len(bboxes_backward) < min_interval:
                first_bbox = bboxes_backward[0] if bboxes_backward else frame_id_to_bbox[template_frame_ids[0]]
                bboxes_backward = [first_bbox] * (min_interval - len(bboxes_backward)) + bboxes_backward
            backward_result = torch.stack(bboxes_backward[:min_interval]).unsqueeze(0)
            gt_frame_ids_backward = gt_frame_ids_backward[:min_interval]
        
        # Process forward sequence
        if len(gt_frame_ids_forward) < future_steps:
            if gt_frame_ids_forward:
                bboxes_forward = [frame_id_to_bbox[fid] for fid in gt_frame_ids_forward if fid in frame_id_to_bbox]
                if bboxes_forward:
                    last_future_bbox = bboxes_forward[-1]
                else:
                    last_future_bbox = frame_id_to_bbox.get(search_end, frame_id_to_bbox[template_frame_ids[0]])
            else:
                last_future_bbox = frame_id_to_bbox.get(search_end, frame_id_to_bbox[template_frame_ids[0]])
                bboxes_forward = []
            
            bboxes_forward.extend([last_future_bbox] * (future_steps - len(bboxes_forward)))
            complete_frame_ids_forward = gt_frame_ids_forward + [gt_frame_ids_forward[-1] if gt_frame_ids_forward else search_end] * (future_steps - len(gt_frame_ids_forward))
            forward_result = torch.stack(bboxes_forward[:future_steps]).unsqueeze(0)
            gt_frame_ids_forward = complete_frame_ids_forward
        else:
            bboxes_forward = [frame_id_to_bbox[fid] for fid in gt_frame_ids_forward if fid in frame_id_to_bbox]
            forward_result = torch.stack(bboxes_forward[:future_steps]).unsqueeze(0)
            gt_frame_ids_forward = gt_frame_ids_forward[:future_steps]
        
        return backward_result, gt_frame_ids_backward, forward_result, gt_frame_ids_forward

    def get_all_frames_optimized(self, dataset, seq_id, template_frame_ids, search_frame_ids, seq_info_dict, min_interval=50, future_steps=5):
        """
        Optimized method to get all required frames (template, search, GT sequence) in a single dataset.get_frames() call
        to minimize I/O overhead and improve data loading performance.
        
        Returns:
            tuple: (template_frames, template_anno, meta_obj_train,
                   search_frames, search_anno, meta_obj_test,
                   gt_sequence_anno_backward, gt_frame_ids_backward,
                   gt_sequence_anno_forward, gt_frame_ids_forward)
        """
        is_video_dataset = dataset.is_video_sequence()
        
        if not is_video_dataset:
            # For image dataset, handle all frames together
            all_frame_ids = list(set(template_frame_ids + search_frame_ids))
            all_frames, all_anno, all_meta = dataset.get_frames(seq_id, all_frame_ids, seq_info_dict)
            
            # Create frame_id to data mapping
            frame_data = {}
            for i, fid in enumerate(all_frame_ids):
                frame_data[fid] = {
                    'frame': all_frames[i],
                    'bbox': all_anno['bbox'][i],
                    'mask': all_anno['mask'][i] if 'mask' in all_anno else None,
                    'meta': all_meta[i] if isinstance(all_meta, list) else all_meta
                }
            
            # Extract template data
            template_frames = [frame_data[fid]['frame'] for fid in template_frame_ids]
            template_bboxes = [frame_data[fid]['bbox'] for fid in template_frame_ids]
            template_masks = [frame_data[fid]['mask'] for fid in template_frame_ids] if frame_data[template_frame_ids[0]]['mask'] is not None else None
            template_anno = {'bbox': template_bboxes}
            if template_masks is not None:
                template_anno['mask'] = template_masks
            meta_obj_train = frame_data[template_frame_ids[0]]['meta']
            
            # Extract search data
            search_frames = [frame_data[fid]['frame'] for fid in search_frame_ids]
            search_bboxes = [frame_data[fid]['bbox'] for fid in search_frame_ids]
            search_masks = [frame_data[fid]['mask'] for fid in search_frame_ids] if frame_data[search_frame_ids[0]]['mask'] is not None else None
            search_anno = {'bbox': search_bboxes}
            if search_masks is not None:
                search_anno['mask'] = search_masks
            meta_obj_test = frame_data[search_frame_ids[0]]['meta']
            
            # Generate GT sequences for image dataset
            bboxes = template_bboxes
            bboxes_backward = bboxes * (min_interval // len(bboxes)) + bboxes[:min_interval % len(bboxes)]
            gt_sequence_anno_backward = torch.stack(bboxes_backward[:min_interval]).unsqueeze(0)
            bboxes_forward = bboxes * (future_steps // len(bboxes)) + bboxes[:future_steps % len(bboxes)]
            gt_sequence_anno_forward = torch.stack(bboxes_forward[:future_steps]).unsqueeze(0)
            
            return (template_frames, template_anno, meta_obj_train,
                   search_frames, search_anno, meta_obj_test,
                   gt_sequence_anno_backward, [],
                   gt_sequence_anno_forward, [])
        
        # For video dataset - collect all frame IDs first
        template_start = min(template_frame_ids)
        search_end = max(search_frame_ids)
        seq_info = dataset.get_sequence_info(seq_id)
        total_frames = seq_info["length"] if "length" in seq_info else len(seq_info["visible"])
        
        # Calculate GT sequence frame IDs
        gt_frame_ids_backward = list(range(template_start, search_end))
        if len(gt_frame_ids_backward) >= min_interval:
            gt_frame_ids_backward = np.linspace(template_start, search_end - 1, min_interval, dtype=int).tolist()
        if len(gt_frame_ids_backward) < min_interval:
            need = min_interval - len(gt_frame_ids_backward)
            new_start = max(0, template_start - need)
            gt_frame_ids_backward = list(range(new_start, search_end))
        
        gt_frame_ids_forward = list(range(search_end, min(search_end + future_steps, total_frames)))
        
        # Collect ALL unique frame IDs needed
        all_needed_frame_ids = set(template_frame_ids + search_frame_ids + gt_frame_ids_backward + gt_frame_ids_forward)
        
        # Handle edge cases for insufficient frames
        if len(gt_frame_ids_backward) < min_interval:
            all_needed_frame_ids.update(template_frame_ids)  # Fallback frames
        if len(gt_frame_ids_forward) < future_steps:
            all_needed_frame_ids.add(search_end)  # Fallback frame
        
        # Single call to get ALL frames
        all_frame_ids_list = sorted(list(all_needed_frame_ids))
        all_frames, all_anno, all_meta = dataset.get_frames(seq_id, all_frame_ids_list, seq_info_dict)
        
        # Create frame_id to data mapping
        frame_data = {}
        for i, fid in enumerate(all_frame_ids_list):
            frame_data[fid] = {
                'frame': all_frames[i],
                'bbox': all_anno['bbox'][i],
                'mask': all_anno['mask'][i] if 'mask' in all_anno else None,
                'meta': all_meta[i] if isinstance(all_meta, list) else all_meta
            }
        
        # Extract template data
        template_frames = [frame_data[fid]['frame'] for fid in template_frame_ids]
        template_bboxes = [frame_data[fid]['bbox'] for fid in template_frame_ids]
        template_masks = [frame_data[fid]['mask'] for fid in template_frame_ids] if frame_data[template_frame_ids[0]]['mask'] is not None else None
        template_anno = {'bbox': template_bboxes}
        if template_masks is not None:
            template_anno['mask'] = template_masks
        meta_obj_train = frame_data[template_frame_ids[0]]['meta']
        
        # Extract search data
        search_frames = [frame_data[fid]['frame'] for fid in search_frame_ids]
        search_bboxes = [frame_data[fid]['bbox'] for fid in search_frame_ids]
        search_masks = [frame_data[fid]['mask'] for fid in search_frame_ids] if frame_data[search_frame_ids[0]]['mask'] is not None else None
        search_anno = {'bbox': search_bboxes}
        if search_masks is not None:
            search_anno['mask'] = search_masks
        meta_obj_test = frame_data[search_frame_ids[0]]['meta']
        
        # Process GT sequences
        # Backward sequence
        if len(gt_frame_ids_backward) < min_interval:
            available_backward_bboxes = [frame_data[fid]['bbox'] for fid in gt_frame_ids_backward if fid in frame_data]
            if available_backward_bboxes:
                first_bbox = available_backward_bboxes[0]
                bboxes_backward = [first_bbox] * (min_interval - len(available_backward_bboxes)) + available_backward_bboxes
            else:
                # Fallback to template bbox
                template_bbox = frame_data[template_frame_ids[0]]['bbox']
                bboxes_backward = [template_bbox] * min_interval
            gt_frame_ids_backward = [gt_frame_ids_backward[0]] * (min_interval - len(gt_frame_ids_backward)) + gt_frame_ids_backward if gt_frame_ids_backward else [template_frame_ids[0]] * min_interval
        else:
            bboxes_backward = [frame_data[fid]['bbox'] for fid in gt_frame_ids_backward[:min_interval]]
            gt_frame_ids_backward = gt_frame_ids_backward[:min_interval]
        
        gt_sequence_anno_backward = torch.stack(bboxes_backward[:min_interval]).unsqueeze(0)
        
        # Forward sequence
        if len(gt_frame_ids_forward) < future_steps:
            available_forward_bboxes = [frame_data[fid]['bbox'] for fid in gt_frame_ids_forward if fid in frame_data]
            if available_forward_bboxes:
                last_bbox = available_forward_bboxes[-1]
                bboxes_forward = available_forward_bboxes + [last_bbox] * (future_steps - len(available_forward_bboxes))
            else:
                # Fallback to search bbox
                search_bbox = frame_data[search_end]['bbox'] if search_end in frame_data else frame_data[search_frame_ids[0]]['bbox']
                bboxes_forward = [search_bbox] * future_steps
            gt_frame_ids_forward = gt_frame_ids_forward + [gt_frame_ids_forward[-1] if gt_frame_ids_forward else search_end] * (future_steps - len(gt_frame_ids_forward))
        else:
            bboxes_forward = [frame_data[fid]['bbox'] for fid in gt_frame_ids_forward[:future_steps]]
            gt_frame_ids_forward = gt_frame_ids_forward[:future_steps]
        
        gt_sequence_anno_forward = torch.stack(bboxes_forward[:future_steps]).unsqueeze(0)
        
        return (template_frames, template_anno, meta_obj_train,
               search_frames, search_anno, meta_obj_test,
               gt_sequence_anno_backward, gt_frame_ids_backward,
               gt_sequence_anno_forward, gt_frame_ids_forward)
