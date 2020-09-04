import cv2
import numpy as np
from numpy import random
import matplotlib.cm as cm

color_dict = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (0, 255, 255),
    'white': (255, 255, 255),
    'pink': (255, 153, 255),
    'orange': (255, 128, 0)
}

colorlist = [v for k,v in color_dict.items()]

def flattenDets(tBoxes, tScores, K = 2, key='', inds=None):
    '''
    '''
    num_classes = len(tBoxes)
    ret = np.zeros((0, K*4 + 2), dtype=np.float32)

    for cls_ind, (clsbox, clsscore) in enumerate(zip(tBoxes, tScores)):
        numdets = clsbox.shape[0]
        if numdets < 1:
            continue
        clsbox = clsbox.reshape((numdets, -1))
        clsscore = clsscore.reshape((numdets, 1))
        clslabel = np.asarray([cls_ind] * numdets)[:, np.newaxis]

        clsdet = np.hstack((clsbox, clslabel, clsscore))
        ret = np.vstack((ret, clsdet))

    if inds is None:
        inds = ret[:,-1].argsort()[::-1]
    ret= ret[inds]
    return {'box'+key:ret[:, :-2], 'label'+key: ret[:, -2], 'score'+key: ret[:, -1], 'inds':inds}

def plotWithBoxesLabelsAndScores(dataset, im_to_plot, detections, threshold=0.3, max_detections=20):
    height, width, channels = im_to_plot[-1].shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    len_tubelet = len(im_to_plot)

    iterable = zip(detections['box'], detections['label'], detections['score'])
    for detection_ind, (box, label, score) in enumerate(iterable):
        if detection_ind >= max_detections:
            break
        
        label = dataset.CLASSES[int(label)]
        
        if (score < threshold):
            continue
        color = color_dict[random.choice(['green', 'blue', 'yellow', 'pink', 'white', 'red', 'orange'])]

        for k in range(len_tubelet):
            boxf = box[k*4 : (k+1)*4].astype(np.int32)
            cv2.rectangle(im_to_plot[k], tuple(boxf[0:2]), tuple(boxf[2:4]), color, 2)
            ### Text box
            txt = '{}: {:.3f}'.format(label, score)
            cv2.rectangle(im_to_plot[k], (boxf[0], int(boxf[1] - 2)), (boxf[2], boxf[1]), (255, 0, 0), 2)
            cv2.putText(im_to_plot[k], txt, (boxf[0], boxf[1]), font, 1, (255, 255, 255), 2)

    return im_to_plot