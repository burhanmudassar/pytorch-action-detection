import os
import pickle
import numpy as np
from .dataset_factory import readsplitfile
from .dataset_factory import Dataset_plus_Torch_Class

JHMDB_CLASSES = (  # always index 0
    'brush_hair',
    'catch',
    'clap',
    'climb_stairs',
    'golf',
    'jump',
    'kick_ball',
    'pick',
    'pour',
    'pullup',
    'push',
    'run',
    'shoot_ball',
    'shoot_bow',
    'shoot_gun',
    'sit',
    'stand',
    'swing_baseball',
    'throw',
    'walk',
    'wave')

JHMDB21_labels = {k+1:v for k,v in enumerate(JHMDB_CLASSES)}
JHMDB21_labels[0] = {'no_activity'} 

class JHMDB21Dataset(Dataset_plus_Torch_Class):
    """UCF24 Action Detection Dataset
    to access input images and target which is annotation
    """

    def __init__(self, name='jhmdb', path='../data/jhmdb/', split=1):
        self.CLASSES = JHMDB_CLASSES
        super(JHMDB21Dataset, self).__init__(name=name, path=path, split=split)

    def get_database(self):
        with open(self.root + 'splitfiles/JHMDB-GT.pkl', 'rb') as fff:
            database = pickle.load(fff, encoding='latin1')
        return database

    def readsplitfile(self):
        trainvideos = self.database['train_videos'][self.split]
        testvideos = self.database['test_videos'][self.split]
        # return trainvideos, testvideos

        return self.trainSet, self.valSet, self.testSet

    def get_train_test_ratios(self):
        ratios = np.asarray([0.28275, 0.29275, 0.27825, 0.254, 0.2925, 0.204, 0.1685, 0.22725, 0.295,
                             0.38025, 0.27825, 0.22725, 0.1905, 0.33425, 0.3035, 0.223, 0.19425, 0.344,
                             0.29125, 0.2555, 0.24225])
        # ratios = np.asarray([1.053,1.131,1.11,0.896,1.091,0.735,0.566,0.841,1.114,1.403,1.041,0.81
        # ,0.602,1.298,1.052,0.816,0.695,1.258,1.208,0.94,0.922])
        return ratios

    def get_video_annotations(self, videoname):
        actidx = self.get_video_act(videoname)
        return self.database['gttubes'][videoname][actidx]

    def get_video_numf(self, videoname):
        return self.database['nframes'][videoname]

    def get_video_act(self, videoname):
        try:
            return list(self.database['gttubes'][videoname].keys())[0]
        except:
            return 9999

    def framepath_rgb(self, vid_name, frame_ind):
        return self.root + 'rgb-images/' + vid_name + '/' + str(frame_ind).zfill(5) + '.png'

    def framepath_brox(self, vid_name, frame_ind):
        return self.root + 'brox-images/' + vid_name + '/' + str(frame_ind).zfill(5) + '.jpg'

    def frame_format(self, v, i):
        return os.path.join(v, "{:0>5}".format(i))

    def get_resolution(self, videoname):
        return [320, 240]

    def load_annotations_to_memory(self):
        print("Loading Annotations to Memory")
        splitfile_train = self.root + 'splitfiles/trainlist{:02d}.txt'.format(self.split)
        splitfile_test = self.root + 'splitfiles/testlist{:02d}.txt'.format(self.split)
        splitfile_val = self.root + 'splitfiles/testlist{:02d}.txt'.format(self.split)

        self.trainSet = readsplitfile(splitfile_train)
        self.testSet = readsplitfile(splitfile_test)
        self.valSet = readsplitfile(splitfile_val)

        self.labels = [v for k, v in JHMDB21_labels.items()]
        self.nlabels = len(self.labels)