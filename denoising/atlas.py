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


class Atlas:
    def __init__(self, atlas_path, atlas_labels):
        self.atlas_path = atlas_path

    @property
    def atlas_labels(self): # TODO не только excel
        roi = pd.read_excel(self.atlas_labels_path, index_col='Index')
        roi_labels = roi.sort_values(by='ID').Label.values
        return {'roi_df': roi, 'roi_labels': roi_labels}

    @property
    def masker(self):
        mask = NiftiLabelsMasker(labels_img=self.atlas_filename,
                                labels=self.atlas_labels['roi_labels'], 
                                memory="nilearn_cache",
                                verbose=2,
                                standardize=True,
                                detrend=True,
                                resampling_target='labels'
                                )
        
        if self.use_cosine is True:
            mask.set_params(high_pass=0.008,
                            low_pass=0.09, 
                            t_r=2.5)
            
        return mask
    
