import warnings
from pathlib import Path
import bids

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.maskers import NiftiLabelsMasker

#warnings.filterwarnings("ignore", category=FutureWarning)


class Denoising:

    def __init__(self, sub_label: list, derivatives_path: str, 
                 atlas_path: str, atlas_labels_path: str,
                 strategy: int, n_compcor=None,
                 use_GSR=False, use_cosine=True,
                 smoothing=None):
        """ 
        
        """
        self.sub_labels = sub_label
        self.derivatives = Path(derivatives_path).as_posix()
        self.atlas_filename = nib.load(Path(atlas_path).as_posix())
        self.atlas_labels_path = Path(atlas_labels_path).as_posix()
        self.int_strategy = strategy
        self.use_cosine = use_cosine
        self.use_GSR = use_GSR
        self.smoothing = smoothing
        self.n_compcor = n_compcor

    @property
    def bids_layout(self):
        return bids.BIDSLayout(
            self.derivatives, validate=False, config=['bids','derivatives'])
    
    
    @property
    def strategy(self):
        strategy_1 = {'strategy': ['motion', 'compcor', 'high_pass'],
                      'motion': 'full',
                      'compcor': 'anat_combined',
                      'n_compcor': 10}
        
        if self.use_GSR:
            strategy_1['strategy'].append('global_signal')
            strategy_1['global_signal'] = 'basic' # TODO выбирать опции
        
        return strategy_1

    #@property
    def get_confounds(self, imgs):
        
        confounds, _ = load_confounds(imgs,
                                    strategy=self.strategy['strategy'],
                                    motion=self.strategy['motion'],
                                    compcor=self.strategy['compcor'], 
                                    n_compcor=self.strategy['n_compcor']
                                    )

        if not self.use_cosine:
            for i in confounds:
                i.drop(['cosine00', 'cosine01', 'cosine02'], axis=1, inplace=True)

        return confounds
    
    
    def denoise(self, sub=None):
        """ 
        
        """
        if sub is None:
            sub = self.bids_layout.get_subjects()

        imgs = self.bids_layout.get(subject=sub,
                        datatype='func', task='rest',
                        desc='preproc',
                        space='MNI152NLin2009cAsym',
                        extension='nii.gz',
                        return_type='file')
        
        conf = self.get_confounds(imgs=imgs)
        denoised_ts = np.empty((len(conf), 120, 426)) # TODO shape
        for i in range(len(conf)):
            denoised_ts[i] = self.masker.fit_transform(imgs[i], confounds=conf[i])

        self.save_outputs(denoised_ts)
        
        return denoised_ts
    
    def save_outputs(self, outputs):
        pass


    def functional_connectivity(self, ts, measure="correlation"):
        connectivity_measure = ConnectivityMeasure(kind=measure)
        fc = connectivity_measure.fit_transform(ts)
        for i in fc:
            np.fill_diagonal(i, 0)

        return fc
