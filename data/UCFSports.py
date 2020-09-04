import os
import pickle
import numpy as np
from .dataset_factory import readsplitfile
from .dataset_factory import Dataset_plus_Torch_Class

UCFSPORTS_CLASSES = (  # always index 0
    'Diving',
    'Golf',
    'Kicking',
    'Lifting',
    'Riding',
    'Run',
    'SkateBoarding',
    'Swing1',
    'Swing2',
    'Walk'
)

UCFSports_labels = {k+1:v for k,v in enumerate(UCFSPORTS_CLASSES)}
UCFSports_labels[0] = {'no_activity'} 

class UCFSportsDataset(Dataset_plus_Torch_Class):
    """UCF24 Action Detection Dataset
    to access input images and target which is annotation
    """

    def __init__(self, name='ucfsports', path='../data/ucfsports/', split=0):
        self.CLASSES = UCFSPORTS_CLASSES
        super().__init__(name=name, path=path, split=split)

    def get_database(self):
        with open(self.root + 'splitfiles/UCFSports-GT.pkl', 'rb') as fff:
            database = pickle.load(fff, encoding='latin1')
        return database

    def readsplitfile(self):
        trainvideos = self.database['train_videos'][self.split]
        testvideos = self.database['test_videos'][self.split]
        return trainvideos, testvideos, testvideos

    def get_train_test_ratios(self):
        ratios = np.asarray([0.54, 0.708, 0.307, 0.476, 0.472, 0.576, 0.552, 0.686, 0.666, 1.617])
        return ratios

    def get_video_annotations(self, videoname):
        actidx = self.get_video_act(videoname)
        return self.database['gttubes'][videoname][actidx]

    def get_video_numf(self, videoname):
        return self.database['nframes'][videoname]

    def get_video_act(self, videoname):
        return list(self.database['gttubes'][videoname].keys())[0]

    def framepath_rgb(self, vid_name, frame_ind):
        return self.root + 'rgb-images/' + vid_name + '/' + str(frame_ind).zfill(6) + '.jpg'

    def framepath_brox(self, vid_name, frame_ind):
        return self.root + 'brox-images/' + vid_name + '/' + str(frame_ind).zfill(6) + '.jpg'

    def frame_format(self, v, i):
        return os.path.join(v, "{:0>5}".format(i))

    def get_resolution(self, videoname):
        return self.database['resolution'][videoname][::-1]

    def load_annotations_to_memory(self):
        print("Loading Annotations to Memory")
        splitfile_train = self.root + 'splitfiles/trainlist{:02d}.txt'.format(self.split)
        splitfile_test = self.root + 'splitfiles/testlist{:02d}.txt'.format(self.split)

        self.trainSet = readsplitfile(splitfile_train)
        self.testSet = readsplitfile(splitfile_test)

        self.labels = [v for k, v in UCFSports_labels.items()]
        self.nlabels = len(self.labels)

        with open(os.path.join(self.root, 'splitfiles/UCFSports-GT.pkl'.format(self.split-1)), 'rb') as fid:
            self.act_dict = pickle.load(fid, encoding='latin')