import os
import pickle
import numpy as np
from .dataset_factory import readsplitfile
from .dataset_factory import Dataset_plus_Torch_Class


UCF101_CLASSES = (  # always index 0
        'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',
        'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing',
        'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling',
        'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog')

UCF101labels = ['no_activity'] + list(UCF101_CLASSES)

class UCF24Dataset(Dataset_plus_Torch_Class):
    """UCF24 Action Detection Dataset
    to access input images and target which is annotation
    """

    def __init__(self, name='UCF101', path='../data/ucf24/', split=1):
        self.CLASSES = UCF101_CLASSES
        super(UCF24Dataset, self).__init__(name=name, path=path, split=split)

    def get_database(self):
        with open(self.root + 'splitfiles/UCF101v2-GT.pkl', 'rb') as fff:
            database = pickle.load(fff, encoding='latin1')
        return database

    def readsplitfile(self):
        trainvideos = self.database['train_videos'][0]
        testvideos = self.database['test_videos'][0]
        
        return self.trainSet, self.valSet, self.testSet

    # def get_train_test_ratios(self):
    #     ratios = np.asarray(
    #         [1.1, 0.8, 4.7, 1.4, 0.9, 2.6, 2.2, 3.0, 3.0, 5.0, 6.2, 2.7, 3.5, 3.1, 4.3, 2.5, 4.5, 3.4, 6.7, 3.6, 1.6,
    #          3.4, 0.6, 4.3])
    #     return ratios
    def get_train_test_ratios(self):
        # ratios = np.asarray(
        #     [0.7405, 0.86375, 4.85875, 1.64425, 0.96625, 2.92775, 4.6245, 3.32275, 2.945, 5.79325, 11.66575, 3.0675,
        #     3.509, 3.445, 1.0205, 2.85025, 5.0185, 3.59975, 7.473, 2.6735, 0.66675, 3.905, 0.56325, 4.64625])
        ratios = np.asarray([0.548,0.5845,2.24025,0.938,0.6965,1.4265,2.59775,1.48925,1.493
        ,1.9515,7.77625,1.55075,2.34625,2.02525,5.3135,1.43425,3.22975,2.1445
        ,5.07775,2.4055,0.97625,4.2625,0.416,3.75825])
        return ratios

    def get_video_annotations(self, videoname):
        # return self.database[videoname]['annotations']

        actidx = self.get_video_act(videoname)
        return self.database['gttubes'][videoname][actidx]

    def get_video_numf(self, videoname):
        # return self.database[videoname]['numf']
        return self.database['nframes'][videoname]

    def get_video_act(self, videoname):
        # return self.database[videoname]['label']
        try:
            return list(self.database['gttubes'][videoname].keys())[0]
        except:
            return 9999

    def get_resolution(self, videoname):
        return [320, 240]

    def framepath_rgb(self, vid_name, frame_ind):
        return self.root + 'rgb-images/' + vid_name + '/' + str(frame_ind).zfill(5) + '.jpg'

    def framepath_brox(self, vid_name, frame_ind):
        return self.root + 'brox-images/' + vid_name + '/' + str(frame_ind).zfill(5) + '.jpg'

    def frame_format(self, v, i):
        return os.path.join(v, "{:0>5}".format(i))

    def load_annotations_to_memory(self):
        print("Loading Annotations to Memory")
        splitfile_train = self.root + 'splitfiles/90trainlist{:02d}.txt'.format(self.split)
        splitfile_val = self.root + 'splitfiles/90vallist{:02d}.txt'.format(self.split)
        splitfile_test = self.root + 'splitfiles/testlist{:02d}.txt'.format(self.split)

        self.trainSet = readsplitfile(splitfile_train)
        self.valSet = readsplitfile(splitfile_val)
        self.testSet = readsplitfile(splitfile_test)

        self.labels = UCF101labels
        self.nlabels = len(self.labels)