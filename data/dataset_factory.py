"""
Dataset utilities - Dataset Class for dataloading
"""

import os
import os.path
import torch
import torch.utils.data as data
import cv2, pickle
import numpy as np
import random

def readsplitfile(splitfile):
    with open(splitfile, 'r') as f:
        temptrainvideos = f.readlines()
    trainvideos = []
    for vid in temptrainvideos:
        vid = vid.rstrip('\n')
        trainvideos.append(vid)
    return trainvideos

class AnnotationTransform(object):
    """
    Same as original
    Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of UCF24's 24 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, CLASSES=None, class_to_ind=None, keep_difficult=False):
        # self.class_to_ind = class_to_ind or dict(
        #     zip(CLASSES, range(len(CLASSES))))
        # self.ind_to_class = dict(zip(range(len(CLASSES)),CLASSES))
        pass

    def __call__(self, bboxs, labels, width, height):
        res = []
        scale = np.asarray([[width, height, width, height]], dtype=np.float32)
        for t in range(len(labels)):
            bbox = bboxs[t,:]
            label = labels[t]
            '''pts = ['xmin', 'ymin', 'xmax', 'ymax']'''
            bndbox = []
            bbox = np.maximum(0, bbox.astype(np.int32) - 1)
            bbox = np.minimum(scale, bbox)
            bbox = bbox.astype(np.float32) / scale
            bndbox.append(bbox)
            bndbox.append(label)
            res += [bndbox]
            # for i in range(4):
            #     cur_pt = max(0,int(bbox[i]) - 1)
            #     scale =  width if i % 2 == 0 else height
            #     cur_pt = min(scale, int(bbox[i]))
            #     cur_pt = float(cur_pt) / scale
            #     bndbox.append(cur_pt)
            # bndbox.append(label)
            # res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


def detection_collate_tubelet(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """

    targets = []
    imgs = []
    image_ids = []
    for sample in batch:
        imgs.append(sample[0])
        # targets.append([torch.FloatTensor(target_frame) for target_frame in sample[1]])
        targets.append([torch.FloatTensor(t) for t in sample[1]])

        image_ids.append(sample[2])

    return torch.stack(imgs, 0), targets, image_ids


class DatasetClass():
    """
    Abstract Class for Data Loading based on the Torch Dataset Class
    """

    def __init__(self, name='UCF101', path='../data/ucf24/', split=1):
        self.name = name
        self.root = path
        self.split = split

        # Load annotations dictionary to memory
        self.load_annotations_to_memory()
        self.database = self.get_database()

    def get_database(self):
        raise NotImplementedError

    def readsplitfile(self):
        raise NotImplementedError

    def get_train_test_ratios(self):
        ratios = np.array([1] * len(self.CLASSES))  # TODO:use this to calculate train/test ratios
        return ratios

    def get_video_annotations(self, videoname):
        raise NotImplementedError

    def get_video_numf(self, videoname):
        raise NotImplementedError

    def get_video_act(self, videoname):
        raise NotImplementedError

    def get_video_tubes(self, videoname):
        annotations = self.get_video_annotations(videoname)
        actidx = self.get_video_act(videoname)
        numf = self.get_video_numf(videoname)

        num_tubes = len(annotations)

        tube_labels = np.zeros((numf ,num_tubes),dtype=np.int16) # check for each tube if present in
        tube_boxes = [[[] for _ in range(num_tubes)] for _ in range(numf)]

        for tubeid, tube in enumerate(annotations):
            # print('numf00', numf, tube['sf'], tube['ef'])
            for frame_id in range(tube.shape[0]): # start of the tube to end frame of the tube
                frame_num = int(tube[frame_id, 0] - 1)
                label = actidx
                # assert actidx == label, 'Tube label and video label should be same'
                box = tube[frame_id, 1:]  # get the box as an array
                box = box.astype(np.float32)
                # Already in x1 y1 x2 y2 format
                try:
                    tube_labels[frame_num, tubeid] = label+1  # change label in tube_labels matrix to 1 form 0
                    tube_boxes[frame_num][tubeid] = box  # put the box in matrix of lists
                except IndexError:
                    print('Out of bounds annotations')
                    print('Video: {} Numf: {:d} and Tube Frame: {:d}'.format(videoname, numf, frame_num))

        return tube_boxes, tube_labels

    def get_frame_path(self, videoname, frame_num):
        if self.input_type == 'rgb':
            return self.framepath_rgb(videoname, frame_num)
        elif self.input_type == 'brox':
            return self.framepath_brox(videoname, frame_num)
        elif self.input_type == 'fusion':
            image_name = [self.framepath_rgb(videoname, frame_num), self.framepath_brox(videoname, frame_num)]
        return image_name

    def framepath_rgb(self, vid_name, frame_ind):
        raise NotImplementedError

    def framepath_brox(self, vid_name, frame_ind):
        raise NotImplementedError

    def get_frame_annotations(self, videoname, frame_num):
        # Get frame annotations for 1-indexed frames?
        tube_boxes, tube_labels = self.get_video_tubes(videoname)

        return np.asarray(tube_boxes[frame_num]), np.asarray(tube_labels[frame_num])

    def get_resolution(self, videoname):
        '''
        return width x height
        '''
        raise NotImplementedError

    def frame_format(self, v, i):
        raise NotImplementedError

    # Merge from ds_utils
    def vlist(self, split):
        if split=='train':
            return self.trainSet
        elif split=='val':
            return self.valSet
        else:
            return self.testSet

    def load_annotations_to_memory(self):
        raise NotImplementedError

    def gtTubes(self, vid_name):
        '''
        :param vid_name: name of video
        :return: tubes corresponding to that video. Dict indexed by class label
                 Each member contains multiple N x 5 arrays [frame index, boxes]
                 box format <x1 y1 x2 y2>
        '''

        return self.database['gttubes'][vid_name]

    # Merge from ds_utils
    def nframes(self, vid_name):
        return self.get_video_numf(vid_name)

    def tubes_unrolled(self, vid_name):
        gttubes=self.gtTubes(vid_name)

        allTubes = []

        tube_ind = 0
        for label, tubes in gttubes.items():
            for tube in tubes:
                lentube = tube.shape[0]
                ind_tube = np.expand_dims(np.asarray([tube_ind] * lentube), 1)
                label_tube = np.expand_dims(np.asarray([label] * lentube), 1)

                final_tube = np.concatenate([ind_tube, label_tube, tube], axis=1)
                allTubes.append(final_tube)
                tube_ind += 1

        return allTubes

class Dataset_plus_Torch_Class(data.Dataset, DatasetClass):

    def __init__(self, name='UCF101', path='../data/ucf24/', split=1):
        DatasetClass.__init__(self, name=name, path=path, split=split)

    @staticmethod
    def fromtorch(cls, root, image_set, transform=None, target_transform=None,
             dataset_name='ucf24', input_type='rgb', full_test=False, num_K=1, split=1, interval=1):
        newcls = cls('', root, split)
        newcls.torchinit(root, image_set, transform, target_transform,
             dataset_name, input_type, full_test, num_K, split, interval)
        return newcls

    def torchinit(self, root, image_set, transform=None, target_transform=None,
             dataset_name='ucf24', input_type='rgb', full_test=False, num_K=1, split=1, interval=1):
        self.splitfile = root + 'splitfiles/trainlist{:02d}.txt'.format(split)
        self.input_type = input_type
        if input_type == 'fusion':
            self._imgpath = [os.path.join(root, i) for i in ['rgb-images', 'brox-images']]
        else:
            self._imgpath = os.path.join(root, self.input_type + '-images')

        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join(root, 'labels/', '%s.txt')
        self.ids = list()
        self.K = num_K
        self.split = split
        self.interval = interval

        annotationlist, video_list = self.make_lists(self, fulltest=full_test)
        self.video_list = video_list
        self.ids = annotationlist

    @staticmethod
    def make_lists(dataset, fulltest=False):
        trainvideos, valvideos, testvideos = dataset.readsplitfile()

        istrain = False
        if dataset.image_set == 'train':
            allvideos = trainvideos
            istrain = True
        elif dataset.image_set == 'val':
            allvideos = valvideos
        elif dataset.image_set == 'test':
            allvideos = testvideos
        else:
            print("Invalid Image Set")
            raise ValueError

        annotationlist = []

        # with open(rootpath + '')

        action_counts = np.zeros(len(dataset.CLASSES), dtype=np.int32)
        ratios = dataset.get_train_test_ratios()
        # ratios = np.ones_like(ratios) #TODO:uncomment this line and line 155, 156 to compute new ratios might be useful for JHMDB21
        video_list = []
        for vid, videoname in enumerate(sorted(allvideos)):
            video_list.append(videoname)
            actidx = dataset.get_video_act(videoname)
            if actidx == 9999:
                print("Annotations are not present so skipping video {}", videoname)
                continue
            if istrain:
                step = ratios[actidx]
            else:
                step =1
            numf = dataset.get_video_numf(videoname)
            lastf = numf - 1
            
            # if videoname not in trainvideos:
                # istrain = False
                # step = max(1, ratios[actidx]) * 1.5  # Taken from AMTNET
            # if fulltest or self.image_set == 'val':
                # step = 1
                # lastf = numf

            annotations = dataset.get_video_annotations(videoname)
            tube_boxes, tube_labels = dataset.get_video_tubes(videoname)

            possible_frame_nums = np.arange(0, numf - (dataset.K - 1), step) # [0, videolength-K]

            if numf - step - 1 not in possible_frame_nums and not istrain:
                possible_frame_nums = np.append(possible_frame_nums, numf - step - 1)

            # print('numf',numf,possible_frame_nums[-1])
            for frame_num in possible_frame_nums:  # loop from start to last possible frame which can make a legit sequence
                frame_num = np.int32(frame_num)
                # Only for fulltest mode we will fix interval to a single range
                if not fulltest:
                    interval = np.random.randint(1, dataset.interval+1)
                else:
                    interval = dataset.interval
                frame_range = dataset.get_frame_range(frame_num, interval)
                # Invalid frame range
                if frame_range[-1] >= numf:
                    continue
                check_tubes = tube_labels[frame_range, :]

                if np.any(np.sum(check_tubes > 0,
                                 axis=0) > dataset.K - 1):  # check if there aren't any semi overlapping tubes
                    sample_boxes = []
                    sample_labels = []
                    tube_Ids = []
                    image_name = dataset.get_frame_path(videoname, frame_num + 1)
                    if dataset.input_type == 'fusion':
                        for img_ in image_name:
                            assert os.path.isfile(img_), 'Image does not exist' + img_
                    else:
                        assert os.path.isfile(image_name), 'Image does not exist' + image_name

                    for tubeid, tube in enumerate(annotations):
                        # if tube_labels[frame_num, tubeid] > 0:
                        tubelet_box = np.asarray([tube_boxes[i][tubeid] for i in frame_range])
                        tubelet_label = np.asarray([tube_labels[i][tubeid] for i in frame_range])
                        tubelet_label = np.unique(tubelet_label)
                        # assert len(tubelet_label) == 1, 'Label for a tube should not change'
                        if len(tubelet_label) != 1: # Skip semi-overlapping tubes
                            continue
                        if tubelet_label == 0:      # Skip negative detection
                            continue
                        sample_boxes.append(tubelet_box)
                        sample_labels.append(tubelet_label)

                    if istrain:  # if it is training video
                        annotationlist.append(
                            [vid, frame_range, np.asarray(sample_labels) - 1, np.asarray(sample_boxes)])
                        for label in sample_labels:
                            action_counts[label - 1] += 1
                    elif dataset.image_set == 'val':
                        annotationlist.append(
                            [vid, frame_range, np.asarray(sample_labels) - 1, np.asarray(sample_boxes)]
                        )
                        for label in sample_labels:
                            action_counts[label - 1] += 1
                    elif dataset.image_set == 'test':  # if test video and has micro-tubes with GT
                        annotationlist.append(
                            [vid, frame_range, np.asarray(sample_labels) - 1, np.asarray(sample_boxes)])
                        for label in sample_labels:
                            action_counts[label - 1] += 1
                elif dataset.image_set == 'test' and not istrain:  # if test video with no ground truth and fulltest is trues
                    annotationlist.append([vid, frame_range, np.asarray([9999]), np.zeros((1, 1, 4))])

        for actidx, act_count in enumerate(
                action_counts):  # just to see the distribution of train and test sets
            print('{:05d} action {:02d} {:s}'.format(act_count,     
                                                        int(actidx),
                                                        dataset.CLASSES[actidx]))

        # newratios = action_counts/1000
        # print('new   ratios', newratios)
        # print('older ratios', ratios)
        print(dataset.image_set, len(annotationlist))
        return annotationlist, video_list

    def __getitem__(self, index):
        im, gt, img_index = self.pull_item(index)

        return im, gt, img_index

    def __len__(self):
        return len(self.ids)

    def get_frame_range(self, frame_num, interval=1):
        '''

        :param frame_num: Return extent of tubelet based on frame_num (0-indexed) and K
        :return:
        '''
        return list(range(frame_num, frame_num + self.K * interval, interval))

    def pull_item(self, index):
        annot_info = self.ids[index]
        frame_range = annot_info[1]
        video_id = annot_info[0]
        videoname = self.video_list[video_id]
        step = 1

        img_names = []
        img_names += [self.get_frame_path(videoname, i+1) for i in frame_range]
        if self.input_type == 'fusion':
            img_names = [i for sublist in img_names for i in sublist]
        try:
            imgs = [cv2.imread(img_).astype(np.float32) for img_ in img_names]
        except AttributeError:
            pass
        height, width, channels = imgs[0].shape
        target_tubelet = self.target_transform(annot_info[3], annot_info[2], width, height)  # Normalizes boxes + Clip
        boxes = np.asarray([t[0][:, :4] for t in target_tubelet])
        labels = np.asarray([t[1] for t in target_tubelet])

        imgs, boxes, labels = self.transform(imgs, boxes, labels)
        imgs = [torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1) for img in imgs]
        targets = [boxes, labels]

        return torch.stack(imgs, 0), targets, index

    def pull_image_tubelet(self, index):
        annot_info = self.ids[index]
        frame_range= annot_info[1]
        video_id = annot_info[0]
        videoname = self.video_list[video_id]
        step = 1

        img_names = []
        img_names += [self.get_frame_path(videoname, i+1) for i in frame_range]
        # Flatten list if fusion mode
        if self.input_type == 'fusion':
            img_names = [i for sublist in img_names for i in sublist]
        imgs = [cv2.imread(img_).astype(np.float32) for img_ in img_names]

        return imgs

import re
def returnValSet(dataset, trainSet):
    # Some groups are divided into clips ... To ensure generalization, remove all clips of one group
    num_total_clips = len(trainSet)
    trainset_seq_list = sorted(trainSet)

    # Create groups - match till last number and then return chars preceding that
    videos_minus_clip = sorted(set([re.match('(.*?)[0-9]+$', seq).group(1) for seq in trainSet]))
    num_groups = len(videos_minus_clip)

    # In-place shuffling of groups
    valset_seq_list_minus_clip = []
    trainset_seq_list_minus_clip = []
    ratio=0.1
    for cls_ in dataset.CLASSES:
        clsVid = [a for a in videos_minus_clip if a.startswith(cls_+'/')]
        random.Random(4).shuffle(clsVid)
        numvalVideos = max(int(len(clsVid)*(ratio)), 1)
        print(cls_, numvalVideos)
        valset_seq_list_minus_clip += clsVid[:numvalVideos]
        trainset_seq_list_minus_clip += clsVid[numvalVideos:]

    # random.Random(4).shuffle(trainset_seq_list_minus_clip)
    # ratio = 0.05

    # Divide groups into train and val set
    # valset_seq_list_minus_clip = trainset_seq_list_minus_clip[int(num_groups*(1-ratio)):]
    # valset_seq_list_minus_clip = trainset_seq_list_minus_clip[::idx]
    # trainset_seq_list_minus_clip = trainset_seq_list_minus_clip[:int(num_groups * (1 - ratio))]

    # Get final list with all clips
    # trainset_seq_list = []
    # valset_seq_list = []
    trainset_seq_list = sorted([a for group in trainset_seq_list_minus_clip for a in trainSet if group in a])
    valset_seq_list = sorted([a for group in valset_seq_list_minus_clip for a in trainSet if group in a])
    

    # valset_seq_list = sorted([seq for seq in trainset_seq_list if  in valset_seq_list_minus_clip])
    # trainset_seq_list = sorted([seq for seq in trainset_seq_list if seq[:idx] not in valset_seq_list_minus_clip])

    assert num_total_clips == len(trainset_seq_list) + len(valset_seq_list)
    assert len(set(trainset_seq_list).intersection(set(valset_seq_list))) == 0

    return trainset_seq_list, valset_seq_list

def CreateValLists(dataset):
    trainSet, valSet = returnValSet(dataset, dataset.trainSet)

    with open(dataset.root + 'splitfiles/90trainlist{:02d}.txt'.format(dataset.split), 'w') as f:
        [f.write(ts+'\n') for ts in trainSet]

    with open(dataset.root + 'splitfiles/90vallist{:02d}.txt'.format(dataset.split), 'w') as f:
        [f.write(vs+'\n') for vs in valSet]
    

if __name__ == '__main__':
    # # d = UCF24_Dataset(path='./ucf24/',split=1)
    # set_ = 'train'
    # ssd_dim = 300
    # means = (104, 117, 123)
    # stds = (0.225, 0.224, 0.229)
    # e = Dataset_plus_Torch_Class.fromtorch(MOVEDetection_Tubelet, './move/', set_, SSDAugmentation(ssd_dim, means, stds),
    #                 AnnotationTransform(), input_type='rgb', num_K=2, split=1)

    # e.gtTubes(e.video_list[0])
    # pass
    # Done with 0.5 ratio
    CreateValLists(UCF24_Dataset('', path='data/ucf24/', split=1, idx=-4))

