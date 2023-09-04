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


class Dataset:

    def __init__(self, sub_label: list, derivatives_path: str):
        """ 
        
        """
        self.sub_labels = sub_label
        self.derivatives = Path(derivatives_path).as_posix()


    @property
    def bids_layout(self):
        return bids.BIDSLayout(
            self.derivatives, validate=False, config=['bids','derivatives'])
    
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
    
    def save_stuff(self):
        pass


    

