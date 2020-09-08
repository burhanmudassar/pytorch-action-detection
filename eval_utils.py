import os
import pickle
import scipy.io as sio
import argparse
import tqdm
from multiprocessing import Process, Queue, Lock
from lib.utils.misc_utils import ensure_dir
from lib.utils.box_utils import *
from lib.utils.ap_utils import pr_to_ap
from data import DatasetBuilder
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_columns', None)

opts = {
    'num_workers' : 8
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoints_dir', type=str, help='directory containing checkpoints')
    parser.add_argument('--eval_iter', type=str, help='rgb and flo detections path [rgb_iter, flo_iter]')

    parser.add_argument('--K', default=1, type=int, required=True, help='Size of subsequences')
    parser.add_argument('--dataset', choices=['ucf24', 'jhmdb', 'ucfsports', 'move'], required=True, help='Dataset to train on')
    parser.add_argument('--path', type=str, required=True, help='Path for the dataset')
    parser.add_argument('--split', default=1, type=int, help='Which split of the dataset to work on')
    parser.add_argument('--redo', dest='redo', help='Redo Evaluation', action='store_true')
    parser.add_argument('--eval_mode', choices=['rgb', 'brox', 'fusion'], required=True, help='Which detections to evaluate')


    args = parser.parse_args()

    _Evaluator = Evaluator(args=args)
    _Evaluator.normal_summarize()

def unit_test_evaluators(dataset, testlist):
    ### Burhan Code to verify that ground truth labels are the same with the ACT -detector and my pytorch evaluator
    countGT = [0 for labels in dataset.labels]
    for iv, vid in enumerate(testlist):
        tubes = dataset.gtTubes(vid)
        for ilabel, label in enumerate(dataset.labels):

            if not ilabel in tubes:
                continue

            for tube in tubes[ilabel]:
                for i in range(tube.shape[0]):
                    countGT[ilabel] += 1
    for il, label in enumerate(dataset.labels[1:]):
        print("{:20s} {:8.2f}".format(label, countGT[il]))


class Evaluator:

    def __init__(self, args):

        self.mode = args.eval_mode
        self.redo = args.redo
        self.detections = None
        self.all_detections = None
        self.actionTubesDir = None
        self.tubes = None
        self.K = args.K

        self.dataset = DatasetBuilder(args.dataset)
        self.dataset = self.dataset(name=args.dataset, path=args.path, split=args.split)

        # Evaluate on test data
        self.testlist = self.dataset.testSet
        eval_itr = int(args.eval_iter)
        self.resultDir = os.path.join(args.checkpoints_dir, args.dataset,
                                      '-'.join([self.mode, str(args.split).zfill(2), str(eval_itr).zfill(6)]))
        ensure_dir(self.resultDir)
        self.get_aggregated_detections()

    def check_if_detections_exist(self):
        if not os.path.isfile(self.resultDir + 'detections.pkl') or self.redo:
            return False
        else:
            return True

    def get_aggregated_detections(self):
        if self.check_if_detections_exist():
            print('Aggregated Detections Found...Loading')
            with open(self.resultDir + 'detections.pkl', 'rb') as fid:
                pickleDict = pickle.load(fid)

            detections = pickleDict['detections']
            all_detections = pickleDict['all_detections']
        else:
            print('Aggregated Detections Not Found ... Generating')
            detections, all_detections = load_frame_detections_parallel_top(self.dataset, self.testlist,
                                                                            self.resultDir, self.K)

            with open(self.resultDir + 'detections.pkl', 'wb') as fid:
                pickle.dump({'detections': detections, 'all_detections': all_detections}, fid, pickle.HIGHEST_PROTOCOL)

        self.detections = detections
        self.all_detections = all_detections

    def frameAP(self, th, testlist, prefix=''):
        frameAP_th = frameAP_offline(self.dataset, testlist, th=th, redo=self.redo, dirname=self.resultDir,
                        alldets=self.detections, prefix=prefix)
        self.dataframe['fAP @ {:.1f}'.format(th)] = frameAP_th

    def frameCLASSIF(self):
        classif = frameCLASSIF_offline(self.dataset, self.testlist, redo=self.redo, dirname=self.resultDir,
                             alldets=self.all_detections)
        self.dataframe['CLASSIF'] = classif

    def frameMABO(self):
        mabo = frameMABO_offline(self.dataset, self.testlist, redo=self.redo, dirname=self.resultDir,
                          alldets=self.detections[:, (0, 1, 4, 5, 6, 7)])
        self.dataframe['MABO'] = mabo

    def BuildTubes(self):
        self.actionTubesDir = os.path.join(self.resultDir,'actionTubes')
        os.makedirs(self.actionTubesDir, exist_ok=True)

        BuildTubes_offline_parallel_top(self.dataset, self.testlist, redo=self.redo, dirname=self.actionTubesDir,
                                        alldets=self.all_detections, K=self.K)

    def videoAP(self, th):
        videoap_th = videoAP(self.dataset, self.testlist, th=th, redo=self.redo, dirname=self.actionTubesDir, alldets=self.tubes)
        self.dataframe['vAP @ {:.2f}'.format(th)] = videoap_th

    def normal_summarize(self):
        self.pd_rows = self.dataset.labels[1:] + ['Mean']
        self.dataframe = pd.DataFrame(index=self.pd_rows, columns=[])

        self.frameAP(th=0.5, testlist=self.testlist)
        self.frameCLASSIF()
        self.frameMABO()
        self.BuildTubes()
        self.tubeloader(vlist=self.testlist, dirname=self.actionTubesDir)
        self.videoAP(th=0.05)
        self.videoAP(th=0.10)
        self.videoAP(th=0.20)
        self.videoAP(th=0.30)
        self.videoAP(th=0.50)
        self.videoAP(th=0.55)
        self.videoAP(th=0.60)
        self.videoAP(th=0.65)
        self.videoAP(th=0.70)
        self.videoAP(th=0.75)
        self.videoAP(th=0.80)
        self.videoAP(th=0.85)
        self.videoAP(th=0.90)
        self.videoAP(th=0.95)

        self.dataframe['vAP @ 0.5:0.95'] = np.mean([self.dataframe['vAP @ {:.2f}'.format(th)].values for th in np.arange(0.5, 1.0, 0.05)], axis=0)
        print(self.dataframe)
        with open(os.path.join(self.resultDir,'summary.txt'), 'w') as fid:
            fid.write(str(self.dataframe))

    def print_summary(self):
        # Print all metrics in a nice and clean tabular form possibly using Pandas
        pass

    def tubeloader(self, vlist, dirname):
        print("Loading all tubes")
        alldets = {ilabel: [] for ilabel in range(self.dataset.nlabels-1)}
        for iv, v in enumerate(vlist):
            tubename = os.path.join(dirname, v + '_tubes.pkl')
            if not os.path.isfile(tubename):
                print("ERROR: Missing extracted tubes " + tubename)
                sys.exit()

            with open(tubename, 'rb') as fid:
                tubes = pickle.load(fid)

            for ilabel in range(self.dataset.nlabels-1):
                ltubes = tubes[ilabel]
                idx = nms3dt(ltubes, 0.3)
                alldets[ilabel] += [(v, ltubes[i][1], ltubes[i][0]) for i in idx]

        self.tubes = alldets


def load_detsfile_kernel(num, d, vlist, K, dirname, queue, lock):
    for iv, v in vlist:
        lock.acquire()
        print("Process:{} {}:Loading detections for {}".format(num, iv, v))
        lock.release()

        w, h = d.get_resolution(v)

        # aggregate the results for each frame
        vdets = {i: np.empty((0, 6), dtype=np.float32) for i in
                 range(1, 1 + d.nframes(v))}  # x1, y1, x2, y2, score, ilabel
        # load results for each starting frame
        for i in range(1, d.nframes(v) + 1):
            resname = os.path.join(dirname, d.frame_format(v, i) + '.pkl')

            if not os.path.isfile(resname):
                print("ERROR: Missing extracted tubelets " + resname)
                # sys.exit()
                continue

            dets_per_frame = load_frame_detections(resname, mode='frame_det')
            # Ignore frames that were not a start frame for any tubelet
            if i <= d.nframes(v)-K+1:
                all_tubelets = load_frame_detections(resname, mode='all_tubelets')
            else:
                all_tubelets = np.zeros((0, 4*K + d.nlabels), dtype=np.float32)
            # Convert to per-frame detections
            detections_to_return = np.zeros((0, 8), dtype=np.float32)
            tubelet_detections = np.zeros((0, 2 + 4*K + d.nlabels), dtype=np.float32)

            if all_tubelets.size != 0:
                tubelet_detections = np.concatenate(
                    (iv * np.ones((all_tubelets.shape[0], 1), dtype=np.float32),
                     i * np.ones((all_tubelets.shape[0], 1), dtype=np.float32),
                     all_tubelets), axis=1)

            if dets_per_frame.size != 0:
                detections_to_return = np.concatenate((iv * np.ones((dets_per_frame.shape[0], 1), dtype=np.float32),
                                                       i * np.ones((dets_per_frame.shape[0], 1), dtype=np.float32),
                                                       dets_per_frame[:, [5, 4, 0, 1, 2, 3]]), axis=1)

            lock.acquire()
            queue.put([detections_to_return, tubelet_detections])
            lock.release()

def load_frame_detections_parallel_top(d, vlist, dirname, K):
    '''

    :param d:           Dataset Class
    :param vlist:       Test list of videos
    :param dirname:     Directory containing detections - framewise
    :return:
    '''
    alldets = [] # list of numpy array with <video_index> <frame_index> <ilabel> <score> <x1> <y1> <x2> <y2>
    alldets_global = []

    num_workers = opts['num_workers']
    iv = range(len(vlist))
    lock = Lock()
    queue = Queue()

    items_per_worker = int(len(vlist) / num_workers)

    processes = []
    for i in range(num_workers):
        if i == num_workers - 1:
            array_to_pass = zip(iv[i*items_per_worker:], vlist[i*items_per_worker:])
        else:
            array_to_pass = zip(iv[i*items_per_worker:(i+1)*items_per_worker],
                                vlist[i*items_per_worker:(i+1)*items_per_worker])


        processes.append(Process(target=load_detsfile_kernel,
                                 args=(i, d, array_to_pass, K, dirname, queue, lock)))
        processes[i].start()

    while 1:
        running = any(p.is_alive() for p in processes)
        while not queue.empty():
            lock.acquire()
            detections, global_detections = queue.get()
            lock.release()
            alldets.append(detections)
            alldets_global.append(global_detections)
        if not running:
            break

        # alldets.append(dets)
    # Creates a 2-D list of all detections with structure N x 8
    # [[ <video_index> <frame_index> <label> <score> <boxes>  ]]
    return np.concatenate(alldets, axis=0), np.concatenate(alldets_global, axis=0)

def BuildTubes_offline_parallel(d, vlist= [], redo=False, dirname='', alldets=np.zeros((0,30),dtype=np.float32), K=2, id=0, lock=Lock()):
    # alldets -> N x <iv> <v> <box4> <scores24>
    for _, v in vlist:
        iv = d.testSet.index(v)
        w,h = d.get_resolution(v)
        outfile = os.path.join(dirname, v + "_tubes.pkl")
        if not os.path.isdir(outfile):
            ensure_dir(outfile)

        if os.path.isfile(outfile) and not redo:
            continue

        lock.acquire()
        print("{}: Processing video {:d}: {:s}".format(id, iv + 1, v))
        lock.release()

        RES = {}
        nframes = d.nframes(v)

        # vdets is dictionary for each video indexed by frame index
        # Each entry is a N x 4 array of boxes
        # vdets = get_class_agnostic_boxes(v)
        # Get detections for the video
        vdets_ind = np.where(alldets[:, 0] == iv)[0]
        vdets = alldets[vdets_ind, :]
        vdets_cols = vdets.shape[1]-2 # 4*K boxes and N+1 scores
        # Split into detections indexed by frame
        VDets = {i: np.empty((0, vdets_cols), dtype=np.float32) for i in range(1, d.nframes(v) + 2 - K)}
        for frameId in range(1, d.nframes(v) + 2 - K):
            det_boxes_frame_ind = np.where(vdets[:, 1] == frameId)[0]
            if len(det_boxes_frame_ind) < 1:
                continue
            det_boxes_frame = vdets[det_boxes_frame_ind, 2:]
            VDets[frameId] = np.concatenate((VDets[frameId], det_boxes_frame),axis=0)

        for ilabel in range(d.nlabels-1):       # 0 is first activity
            FINISHED_TUBES = []
            CURRENT_TUBES = []  # tubes is a list of tuple (frame, lstubelets)

            def tubescore(tt):
                return np.mean(np.array([tt[i][1][-1] for i in range(len(tt))]))

            for frame in range(1, d.nframes(v) + 2 - K):
                # load boxes of the new frame and do nms while keeping Nkeep highest scored

                boxes = VDets[frame][:,range(4*K)] * np.repeat(np.asarray([[w, h, w, h]]), K, axis=0).reshape(1, K*4)
                scores = VDets[frame][:, [4*K + 1 + ilabel]]

                # scores_gt_th = np.where(scores > 0.2)[0]
                # boxes = boxes[scores_gt_th, :]
                # scores = scores[scores_gt_th, :]

                ltubelets = np.concatenate([boxes, scores], axis=1)
                idx = nms_tubelets(ltubelets, 0.3, top_k=10)
                ltubelets = ltubelets[idx, :]

                # just start new tubes
                if frame == 1:
                    for i in range(ltubelets.shape[0]):
                        CURRENT_TUBES.append([(1, ltubelets[i, :])])
                    continue

                # sort current tubes according to average score
                avgscore = [tubescore(t) for t in CURRENT_TUBES]
                argsort = np.argsort(-np.array(avgscore))
                CURRENT_TUBES = [CURRENT_TUBES[i] for i in argsort]

                # loop over tubes
                finished = []
                for it, t in enumerate(CURRENT_TUBES):
                    # compute ious between the last box of t and ltubelets
                    last_frame, last_tubelet = t[-1]
                    offset = frame - last_frame
                    if offset < K:
                        nov = K - offset
                        ious = sum([iou2d(ltubelets[:, 4 * iov:4 * iov + 4],
                                          last_tubelet[4 * (iov + offset):4 * (iov + offset + 1)]) for iov in
                                    range(nov)]) / float(nov)
                    else:
                        ious = iou2d(ltubelets[:, :4], last_tubelet[4 * K - 4:4 * K])

                    valid = np.where(ious >= 0.2)[0]

                    if valid.size > 0:
                        # take the one with maximum score
                        idx = valid[np.argmax(ltubelets[valid, -1])]
                        CURRENT_TUBES[it].append((frame, ltubelets[idx, :]))
                        ltubelets = np.delete(ltubelets, idx, axis=0)
                    else:
                        # k parameter for keeping tube without any detection
                        if offset >= 1:
                            finished.append(it)

                # finished tubes that are done
                for it in finished[::-1]:  # process in reverse order to delete them with the right index
                    FINISHED_TUBES.append(CURRENT_TUBES[it][:])
                    del CURRENT_TUBES[it]

                # start new tubes
                for i in range(ltubelets.shape[0]):
                    CURRENT_TUBES.append([(frame, ltubelets[i, :])])

            # all tubes are not finished
            FINISHED_TUBES += CURRENT_TUBES

            # build real tubes
            output = []
            for t in FINISHED_TUBES:
                score = tubescore(t)

                # just start new tubes
                if score < 0.01:
                    continue

                beginframe = t[0][0]
                endframe = t[-1][0] + K - 1
                length = endframe + 1 - beginframe

                # delete tubes with short duraton - why????
                if length < 15:
                    continue

                # build final tubes by average the tubelets
                out = np.zeros((length, 6), dtype=np.float32)
                out[:, 0] = np.arange(beginframe, endframe + 1)
                n_per_frame = np.zeros((length, 1), dtype=np.int32)
                for i in range(len(t)):
                    frame, box = t[i]
                    for k in range(K):
                        out[frame - beginframe + k, 1:5] += box[4 * k:4 * k + 4]
                        out[frame - beginframe + k, -1] += box[-1]
                        n_per_frame[frame - beginframe + k, 0] += 1
                out[:, 1:] /= n_per_frame
                output.append((out, score))

            RES[ilabel] = output

        with open(outfile, 'wb') as fid:
            pickle.dump(RES, fid)


def BuildTubes_offline_parallel_top(d, vlist= [], redo=False, dirname='', alldets=np.zeros((0,30),dtype=np.float32), K=2):
    num_workers = opts['num_workers']
    iv = range(len(vlist))
    lock = Lock()

    items_per_worker = int(len(vlist) / num_workers)

    processes = []
    for i in range(num_workers):
        if i == num_workers - 1:
            array_to_pass = zip(iv[i*items_per_worker:], vlist[i*items_per_worker:])
        else:
            array_to_pass = zip(iv[i*items_per_worker:(i+1)*items_per_worker],
                                vlist[i*items_per_worker:(i+1)*items_per_worker])


        processes.append(Process(target=BuildTubes_offline_parallel, args=(d, array_to_pass, redo, dirname, alldets, K, i, lock)))
        processes[i].start()

    while 1:
        running = any(p.is_alive() for p in processes)
        # while not queue.empty():
        #     detections, global_detections = queue.get()
        #     alldets.append(detections)
        #     alldets_global.append(global_detections)
        if not running:
            break


def unroll_classIndexedDets(detections):
    '''

    :param detections: An list of length 'num_classes' with dimensions 'N x 5' '<x1 y1 x2 y2> score
    :return: an array of detections N x 6 label score x1 y1 x2 y2
    '''
    detections_unrolled = np.empty((0,6),dtype=np.float32)
    for class_ind, class_dets in enumerate(detections):
        class_labels = np.array([class_ind]*class_dets.shape[0])[:, np.newaxis]
        class_dets = class_dets[:,(4, 0, 1, 2 ,3)]
        class_dets = np.hstack((class_labels, class_dets))
        detections_unrolled = np.concatenate((detections_unrolled, class_dets), axis=0)

    detections_unrolled = detections_unrolled[detections_unrolled[:,1].argsort()[::-1]]
    return detections_unrolled


def frameAP_offline(d, vlist=[], th=0.5, redo=False, dirname='', alldets=np.zeros((0,8),dtype=np.float32), prefix='', index_switch=False):
    eval_file = os.path.join(dirname, "frameAP{:g}_{:s}.pkl".format(th,prefix))
    ensure_dir(eval_file)

    if isinstance(vlist, str):
        vlist = [vlist]

    if os.path.isfile(eval_file) and not redo:
        with open(eval_file, 'rb') as fid:
            res = pickle.load(fid)
    else:
        # Test Function to return ground truth as detections - to get 100 % frame AP
        def debugFrameAP(vlist):
            arraydets = []
            for _, v in enumerate(vlist):
                iv = d.testSet.index(v)
                tubes = d.gtTubes(v)
                for ilabel, label in enumerate(d.labels):

                    if not ilabel in tubes:
                        continue

                    for tube in tubes[ilabel]:
                        for i in range(tube.shape[0]):
                            temp = [iv, tube[i, 0], ilabel, 1.0]
                            temp.extend(tube[i, 1:].tolist())
                            arraydets.append(temp)

            return np.array(arraydets)

        res = {}

        ## Filter out detections not belonging to the videos we are testing
        iv_inds = [d.testSet.index(v) for v in vlist]
        alldets = alldets[np.isin(alldets[:, 0], iv_inds), :]


        # compute AP for each class
        for ilabel, label in enumerate(d.labels[1:]):
            print("Computing AP for class {}:{}".format(ilabel, label))
            # detections of this class
            if index_switch: # Detections are 1-indexed
                detections = alldets[alldets[:, 2] == ilabel+1, :]
            else: # Detections are 0-indexed
                detections = alldets[alldets[:, 2] == ilabel, :]
            # load ground-truth of this class
            # gt is a Dictionary indexed by two keys i.e. [ <video_index> <frame_index> ]. Each member is N x 4
            gt = {}
            for _, v in enumerate(vlist):
                iv = d.testSet.index(v)
                tubes = d.gtTubes(v)

                if not ilabel in tubes:
                    continue

                # Iterate over all tubes
                for tube in tubes[ilabel]:
                    # Each tube is Nx5 [<frame_index> <box>]
                    for i in range(tube.shape[0]):
                        # Create index for gt
                        k = (iv, int(tube[i, 0]))
                        if not k in gt:
                            gt[k] = []
                        gt[k].append(tube[i, 1:5].tolist())

            for k in gt:
                gt[k] = np.array(gt[k])

            # pr will be an array containing precision-recall values
            gt_count = sum([g.shape[0] for g in gt.values()])  # false negatives
            fp = 0  # false positives
            tp = 0  # true positives
            if gt_count == 0:
                pr = np.zeros((1,2), dtype=np.float32)
                pr[0, 0] = 0.0
                pr[0, 1] = 0.0
                res[label] = pr
                continue
            pr = np.empty((detections.shape[0] + 1, 2), dtype=np.float32)  # precision,recall
            pr[0, 0] = 1.0  # 100 % precision
            pr[0, 1] = 0.0  # 0 % Recall

            for i, j in enumerate(np.argsort(-detections[:, 3])):
                k = (int(detections[j, 0]), int(detections[j, 1]))
                box = detections[j, 4:8]
                ispositive = False
                if k in gt:
                    ious = iou2d(gt[k], box)
                    amax = np.argmax(ious)
                    if ious[amax] >= th:
                        ispositive = True
                        gt[k] = np.delete(gt[k], amax, 0)

                        if gt[k].size == 0:
                            del gt[k]
                if ispositive:
                    tp += 1
                else:
                    fp += 1

                # if fn == 0:
                #     pr[i + 1, 0] = 1.0
                # else:
                pr[i + 1, 0] = float(tp) / float(tp + fp)
                if gt_count == 0:
                    pr[i + 1, 1] = 0.0
                else:
                    pr[i + 1, 1] = float(tp) / float(gt_count)

            res[label] = pr
        # save results
        with open(eval_file, 'wb') as fid:
            pickle.dump(res, fid)

    # display results
    # Use trapezoid integration formula to convert pr_ap values to AP
    ap = 100 * np.array([pr_to_ap(res[label]) for label in d.labels[1:]])
    # ap = 100 * np.array([voc_ap(res[label][:,1], res[label][:,0]) for label in d.labels[1:]])
    print("frameAP")

    for il, label in enumerate(d.labels[1:]):
        print("{:20s} {:8.2f}".format(label, ap[il]))

    print("{:20s} {:8.2f}".format("mAP", np.mean(ap)))
    print("")

    ap = np.concatenate((ap, [np.mean(ap)]))
    return ap


def frameMABO_offline(d, vlist = [], redo=False, dirname='', alldets = np.zeros((0,6), dtype=np.float32)):
    ### alldets is N x 6 array <video_index> <frame_index> <boxes (x1 y1 x2 y2)>

    eval_file = os.path.join(dirname, "frameMABO.pkl")

    if os.path.isfile(eval_file) and not redo:
        with open(eval_file, 'rb') as fid:
            BO = pickle.load(fid)
    else:
        BO = {l: [] for l in d.labels[1:]}  # best overlap

        for iv, v in enumerate(vlist):
            print("{}:Computing BO for {}".format(iv,v))
            gt = d.gtTubes(v)
            w,h = d.get_resolution(v)

            ### vdets is dictionary for each video indexed by frame index
            ### Each entry is a N x 4 array of boxes
            # vdets = get_class_agnostic_boxes(v)
            # Get detections for the video
            vdets_ind = np.where(alldets[:,0] == iv)[0]
            vdets = alldets[vdets_ind, :]
            # Split into detections indexed by frame
            frameDets = {i: np.empty((0, 4), dtype=np.float32) for i in range(1, 1 + d.nframes(v))}
            for frameId in range(1, 1 + d.nframes(v)):
                det_boxes_frame_ind = np.where(vdets[:,1] == frameId)[0]
                if len(det_boxes_frame_ind) < 1:
                    continue
                det_boxes_frame = vdets[det_boxes_frame_ind,2:]
                frameDets[frameId] = det_boxes_frame

            vdets = frameDets

            # for each frame
            for i in range(1, 1 + d.nframes(v)):
                for ilabel in gt:
                    label = d.labels[ilabel+1]
                    for t in gt[ilabel]:
                        # the gt tube does not cover frame i
                        if not i in t[:, 0]:
                            continue

                        gtbox = t[t[:, 0] == i, 1:5]  # box of gt tube at frame i

                        if vdets[i].size == 0:  # we missed it
                            BO[label].append(0)
                            continue

                        ious = iou2d(vdets[i], gtbox)
                        BO[label].append(np.max(ious))
            # save file
            with open(eval_file, 'wb') as fid:
                pickle.dump(BO, fid)

    # print MABO results
    ## Add 0 for empty categories
    for la in d.labels[1:]:
        if len(BO[la]) == 0:
            BO[la] = [0]

    ABO = {la: 100 * np.mean(np.array(BO[la])) for la in d.labels[1:]}  # average best overlap

    for la in d.labels[1:]:
        print("{:20s} {:6.2f}".format(la, ABO[la]))

    ABO_values = np.array([ABO[k] for k in d.labels[1:]]).astype(np.int32)
    MABO = np.mean(ABO_values)
    print("{:20s} {:6.2f}".format("MABO", MABO))
    print("")

    ABO_values = np.concatenate((ABO_values, [MABO]))
    return ABO_values


def frameCLASSIF_offline(d, vlist= [], redo=False, dirname='', alldets=np.zeros((0,30), dtype=np.float32), K=2):
    eval_file = os.path.join(dirname, "frameCLASSIF.pkl")

    if os.path.isfile(eval_file) and not redo:
        with open(eval_file, 'rb') as fid:
            CLASSIF = pickle.load(fid)
    else:
        vlist = vlist
        CORRECT = [0 for ilabel in range(d.nlabels-1)]
        TOTAL = [0 for ilabel in range(d.nlabels-1)]

        for iv, v in enumerate(vlist):
            w,h = d.get_resolution(v)
            print("{}:Computing CLASSIF for {}".format(iv, v))
            nframes = d.nframes(v)

            vdets_ind = np.where(alldets[:,0] == iv)[0]
            vdets = alldets[vdets_ind, :]
            # Split into detections indexed by frame
            frameDets = {i: np.empty((0, 4 + d.nlabels - 1), dtype=np.float32) for i in range(1, 1 + d.nframes(v))}
            for frameId in range(1, 1 + d.nframes(v)):
                det_boxes_frame_ind = np.where(vdets[:,1] == frameId)[0]
                if len(det_boxes_frame_ind) < 1:
                    continue
                det_boxes_frame = vdets[det_boxes_frame_ind,2:]
                frameDets[frameId] = det_boxes_frame

            VDets = frameDets


            # iterate over ground-truth
            tubes = d.gtTubes(v)
            for ilabel in tubes:
                for g in tubes[ilabel]:
                    for i in range(g.shape[0]):
                        frame = int(g[i, 0])

                        # just in case a tube is longer than the video
                        if frame > nframes:
                            continue

                        gtbox = g[i, 1:5]
                        scores = np.zeros((d.nlabels-1,), dtype=np.float32)

                        # average the score over the K frames from all start frames (i.e. for K=2 sf=8 and sf=9
                        # both predicted
                        for sf in range(max(1, frame - K + 1), min(nframes - K + 1, frame) + 1):
                            boxes = VDets[sf][:, 4 * (frame - sf):4 * (frame - sf) + 4] *[[w,h,w,h]]
                            overlaps = iou2d(boxes, gtbox)
                            if len(overlaps) > 0:
                                scores += np.sum(VDets[sf][overlaps >= 0.7, 4 * K + 1:], axis=0)

                        # check classif
                        if np.argmax(scores) == ilabel:
                            CORRECT[ilabel] += 1

                        TOTAL[ilabel] += 1

        CLASSIF = [float(CORRECT[ilabel]) / float(TOTAL[ilabel]) if TOTAL[ilabel] !=0 else 0 for ilabel in range(d.nlabels-1)]

        with open(eval_file, 'wb') as fid:
            pickle.dump(CLASSIF, fid)

    CLASSIF = np.array([100 * c_ for c_ in CLASSIF])
    # print classif results
    for il, la in enumerate(d.labels[1:]):
        print("{:20s} {:6.2f}".format(la, CLASSIF[il]))

    print("{:20s} {:6.2f}".format("CLASSIF", np.mean(CLASSIF)))
    print("")

    CLASSIF = np.concatenate((CLASSIF, [np.mean(CLASSIF)]))
    return CLASSIF


def videoAP(d, vlist=[], th=0.5, redo=False, dirname='', alldets={}):
    eval_file = os.path.join(dirname, "videoAP{:g}.pkl".format(th))

    if os.path.isfile(eval_file) and not redo:
        with open(eval_file, 'rb') as fid:
            res = pickle.load(fid)
    else:
        # compute AP for each class
        res = {}
        for ilabel in range(d.nlabels-1):
            detections = alldets[ilabel]
            # load ground-truth
            gt = {}
            for v in vlist:
                tubes = d.gtTubes(v)

                if not ilabel in tubes:
                    continue

                gt[v] = tubes[ilabel]

                # if len(gt[v]) == 0:
                #     del gt[v]

            for k in gt:
                gt[k] = np.array(gt[k])

            # precision,recall
            pr = np.empty((len(detections) + 1, 2), dtype=np.float32)
            pr[0, 0] = 1.0
            pr[0, 1] = 0.0

            fn = sum([len(g) for g in gt.values()])  # false negatives
            fp = 0  # false positives
            tp = 0  # true positives

            for i, j in enumerate(np.argsort(-np.array([dd[1] for dd in detections]))):
                v, score, tube = detections[j]
                ispositive = False

                if v in gt:
                    ious = [iou3dt(g, tube) for g in gt[v]]
                    amax = np.argmax(ious)
                    if ious[amax] >= th:
                        ispositive = True
                        gt[v] = np.delete(gt[v], amax, 0)
                        # del gt[v][amax]
                        if gt[v].size == 0:
                            del gt[v]

                if ispositive:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1

                pr[i + 1, 0] = float(tp) / float(tp + fp)
                if tp + fn == 0:
                    pr[i + 1, 1] = 0.0
                else:
                    pr[i + 1, 1] = float(tp) / float(tp + fn)

            res[d.labels[ilabel+1]] = pr

        # save results
        with open(eval_file, 'wb') as fid:
            pickle.dump(res, fid)

    # display results
    ap = 100 * np.array([pr_to_ap(res[label]) for label in d.labels[1:]])
    print("videoAP {:0.2f}".format(th))
    for il, label in enumerate(d.labels[1:]):
        print("{:20s} {:8.2f}".format(label, ap[il]))
    mAP = np.mean(ap)
    print("{:20s} {:8.2f}".format("mAP", mAP))
    print("")
    ap = np.concatenate((ap, [mAP]))
    return ap

def load_frame_detections(fname, mode='frame_det'):
    '''
    Open pickle file and either return per-frame NMSed detections
    OR return all tubelets ranked according to max non background score (3D NMSed at 0.7)
    :param fname: pickle filename
    :param mode:  mode
    :return:
    '''
    fid = open(fname, 'rb')
    if mode == 'frame_det':
        # list of numpy array with <x1> <y1> <x2> <y2> <score> <ilabel>
        reference = np.zeros((0, 6), dtype=np.float32)
        while 1:
            try:
                reference = np.concatenate([reference, pickle.load(fid)['frame_det']], axis=0)
            except EOFError:
                break
            except KeyError:
                continue
    elif mode == 'all_tubelets':
        reference = pickle.load(fid)['all_tubelets']

    return reference


if __name__ == '__main__':
    main()

