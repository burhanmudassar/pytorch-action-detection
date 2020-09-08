import os
import torch
import numpy as np
import pickle
from lib.modeling.layers.box_utils import nms, nms3d

# Convert location and confidence to a det structure - Rewrite using save_framedets_clsmajor
def convert_locconf_to_dets(loc, conf_scores, conf_thresh, num_classes, nms_thresh, top_k, return_ids=False):
    '''
    Takes in decoded boxes and class scores and returns intra-class NMSed + top_k + ClassMajor Indexed Dets
    :param loc:
    :param conf_scores:
    :param conf_thresh:
    :param num_classes:
    :param nms_thresh:
    :param top_k:
    :return:
    '''
    output = torch.zeros(num_classes, top_k, 5)
    if return_ids:
        ids = torch.zeros(num_classes, top_k, 1).long()

    for cl_ in range(1, num_classes):
        c_mask = conf_scores[:, cl_].gt(conf_thresh).nonzero().view(-1)
        if c_mask.size(0) == 0:
            continue
        scores = conf_scores[c_mask, cl_]
        boxes = loc[c_mask, :]

        new_boxes = boxes * 300
        ids_nms = gpu_nms(torch.cat((new_boxes, scores.unsqueeze(1)), 1).clone().cpu().numpy(), nms_thresh)

        if len(ids_nms) > top_k:
            count = top_k
        else:
            count = len(ids_nms)

        output[cl_, :count] = torch.cat((scores[ids_nms[:count]].unsqueeze(1), boxes[ids_nms[:count]]), 1)
        if return_ids:
            ids[cl_, :count] = torch.Tensor(ids_nms[:count]).unsqueeze(1).long()

    if return_ids:
        return output, ids
    else:
        return output

# Save the tubelets with all class scores for a clip
def save_tubelets_nms3d(decoded_boxes, conf_scores, output_dir, videoname, frame_range_1):
    video_dir = os.path.join(output_dir, videoname)
    if not os.path.isdir(video_dir):
        os.makedirs(video_dir)

    c_mask = conf_scores[:, 1:].max(1)[0].gt(0)
    l_mask = c_mask.unsqueeze(-1).unsqueeze(-1).expand_as(decoded_boxes)
    tubelet_boxes = decoded_boxes[l_mask]
    tubelet_scores = conf_scores[c_mask]

    tubelet_boxes = tubelet_boxes.view(-1, decoded_boxes.size(1) * decoded_boxes.size(2))
    max_scores = tubelet_scores[:, 1:].max(1)[0]
    ids, counts = nms3d(tubelet_boxes, max_scores, 0.9, 300)
    tubelet_boxes = tubelet_boxes[ids[:counts]]
    tubelet_scores = tubelet_scores[ids[:counts], :]

    tubelet_det = torch.cat([tubelet_boxes, tubelet_scores], dim=1).cpu().numpy()

    # TODO: Bug here. This only writes to frame_range[0] but ignores frame[1] and onwards
    # Actual bug: For frame 15, it would rewrite the pickle file for frame 15 but would ignore 16
    # 16 would get appended over and over again in the next loop
    output_file_name = video_dir + '/{:05d}.pkl'.format(int(frame_range_1[0]))
    with open(output_file_name, 'wb') as fid:
        pickle.dump({'all_tubelets': tubelet_det}, fid)

# Save the frame detections indexed by class in frame_range
def tubeletNMS(decoded_boxes, conf_scores, width, height, conf_thresh, nms_thresh, topk, num_classes, return_inds = False):
    '''
    Take in decoded tubelets and conf_scores and apply per-class 3D NMS
    K : length of tubelet
    L : number of boxes/scores
    input: decoded_boxes -> normalized boxes L x K x 4
            conf_scores -> L x 1
            width -> image
            height -> image
            num_classes
            conf_thresh -> threshold for score
            nms_thres -> threshold for 3D NMS
            topk -> max boxes to keep
    '''
    tubelet_boxes = []
    tubelet_scores = []
    tubelet_softmax = []
    # tubelet_keys = []
    # keys_range = torch.arange(0, conf_scores.size(0), step=1).long()
    for cl_ind in range(1, num_classes):
        scores = conf_scores[:, cl_ind].squeeze()
        c_mask = scores.gt(conf_thresh)  # greater than minmum threshold
        scores = scores[c_mask]
        # keys = keys_range[c_mask]
        softmax_scores = conf_scores[c_mask]
        if scores.numel() == 0:
        # if scores.size(0) == 0:
            tubelet_boxes.append(np.zeros((0, decoded_boxes.size(1), 4), dtype=np.float32))
            tubelet_scores.append([])
            # tubelet_keys.append([])
            tubelet_softmax.append(np.zeros((0, conf_scores.size(1)), dtype=np.float32))
            continue

        boxes = decoded_boxes.clone()
        l_mask = c_mask.unsqueeze(-1).unsqueeze(-1).expand_as(boxes)
        boxes = boxes[l_mask].view(-1, boxes.size(1) * boxes.size(2))
        # idx of highest scoring and non-overlapping boxes per class
        ids, counts = nms3d(boxes, scores, nms_thresh, topk)  # idsn - ids after nms
        scores = scores[ids[:counts]].cpu().numpy()
        # keys = keys[ids[:counts]].cpu().numpy()
        softmax_scores = softmax_scores[ids[:counts]].cpu().numpy()
        boxes = boxes.view(boxes.size(0), -1, 4)
        boxes = boxes[ids[:counts]].cpu().numpy()
        boxes *= [[[width, height, width, height]]]

        for ik in range(boxes.shape[0]):
            boxes[ik, :, 0] = np.maximum(0, boxes[ik, :, 0])
            boxes[ik, :, 2] = np.minimum(width, boxes[ik, :, 2])
            boxes[ik, :, 1] = np.maximum(0, boxes[ik, :, 1])
            boxes[ik, :, 3] = np.minimum(height, boxes[ik, :, 3])

        tubelet_boxes.append(boxes)
        tubelet_scores.append(scores)
        # tubelet_keys.append(keys)
        tubelet_softmax.append(softmax_scores)

    if return_inds:
        return tubelet_boxes, tubelet_scores, tubelet_softmax
    else:
        return tubelet_boxes, tubelet_scores, None


def storeTubeletDetections(videoname, frame_range_1, tubelet_boxes, tubelet_scores, det_boxes):
    '''
        Stores boxes and scores from a tubelet into a global array indexed by videoname + frame_number
        Input:
            tubelet_boxes: list of length N containing L x 2 x 4 boxes
            tubelet_scores: list of length N containing L x 1 scores
            frame_range_1: frame number
            videoname : name of video
            det_boxes: dict containing global results -> [classIdx][videoname][frame_num][boxes, scores]
    '''
    num_classes = len(det_boxes) + 1
    for cl_ind in range(1, num_classes):
        for f_idx, f_ in enumerate(frame_range_1):
            if det_boxes[cl_ind - 1].get(videoname, None) is None:
                det_boxes[cl_ind - 1][videoname] = {}
            if det_boxes[cl_ind - 1][videoname].get(f_, None) is None:
                det_boxes[cl_ind - 1][videoname][f_] = [np.zeros((0, 4), dtype=np.float32), np.zeros((0), dtype=np.float32)]

            boxes_cls = tubelet_boxes[cl_ind-1]
            scores_cls = tubelet_scores[cl_ind-1]
            if len(scores_cls) < 1:
                continue
            # Store in det boxes according to frame number
            det_boxes[cl_ind - 1][videoname][f_][0] = \
                np.concatenate((boxes_cls[:, f_idx, :], det_boxes[cl_ind - 1][videoname][f_][0]), axis=0)
            det_boxes[cl_ind - 1][videoname][f_][1] = \
                np.concatenate((scores_cls[:], det_boxes[cl_ind - 1][videoname][f_][1]), axis=0)

# Get the frame ground truth and store gt_boxes and return frame range for that frame number
def get_frameGT(val_dataset, videoname, frame_range, gt_boxes):
    frame_tubes = [val_dataset.get_frame_annotations(videoname, f_) for f_ in frame_range]
    frame_range_1 = [f + 1 for f in frame_range]
    for f_idx, f_ in enumerate(frame_range_1):
        if gt_boxes.get(videoname, None) is None:
            gt_boxes[videoname] = {}
        if gt_boxes[videoname].get(f_, None) is None:
            gt_boxes[videoname][f_] = [frame_tubes[f_idx][0], frame_tubes[f_idx][1] - 1]
        else:
            pass
    return frame_range_1

# Re run per-frame NMS as sliding window might save redundant detections for the same frame (if not FP will increase)
def final_frameNMS(det_boxes, save_results, output_dir, nms_thresh, topk):
    for cls_ind, _ in enumerate(det_boxes):
        for video in det_boxes[cls_ind].keys():
            for nf in det_boxes[cls_ind][video].keys():
                boxes = det_boxes[cls_ind][video][nf][0]
                scores = det_boxes[cls_ind][video][nf][1]

                if len(scores) == 0:
                    continue

                ids, counts = nms(torch.from_numpy(boxes), torch.from_numpy(scores), nms_thresh, topk)
                boxes = boxes[ids[:counts], :]
                scores = scores[ids[:counts]]

                # Make sure boxes are Nx4
                if len(boxes.shape) == 1:
                    boxes = np.expand_dims(boxes, axis=0)
                if len(scores.shape) == 0:
                    scores = np.expand_dims(scores, axis=0)

                det_boxes[cls_ind][video][nf][0] = boxes
                det_boxes[cls_ind][video][nf][1] = scores

                if save_results:
                    label = np.expand_dims(np.asarray([cls_ind] * boxes.shape[0]), axis=1)
                    frame_det = np.concatenate((boxes, np.expand_dims(scores, axis=1), label), axis=1)
                    # Store frame detection here
                    video_dir = os.path.join(output_dir, video)
                    output_file_name = video_dir + '/{:05d}.pkl'.format(nf)
                    with open(output_file_name, 'ab') as fid:
                        pickle.dump({'frame_det': frame_det}, fid)