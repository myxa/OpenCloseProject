from pathlib import Path
import os
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_aal


class Atlas:
    def __init__(self, atlas_name, atlas_path=None):

        if atlas_name not in ['HCPex', 'Schaefer200', 'AAL']:
            raise NotImplementedError('Available atlases: HCPex, Schaefer200, AAL')

        self.atlas_name = atlas_name
        self.atlas_labels_path = Path('../atlas/')

        if atlas_name != 'AAL':
            self.atlas_path = Path(atlas_path).as_posix()
        else:
            self.atlas = fetch_atlas_aal(data_dir=self.atlas_labels_path)
            self.atlas_path = self.atlas['maps']


    @property
    def atlas_labels(self): 

        if self.atlas_name == 'HCPex':
            roi = pd.read_excel(os.path.join(self.atlas_labels_path, 'HCPex_sorted_networks_names.xlsx', 
                                index_col='Index'))

            roi_labels = roi.sort_values(by='ID').Label.values

        elif self.atlas_name == 'Schaefer200':
            roi = pd.read_csv(os.path.join(self.atlas_labels_path, 'schaefer200_atlas.txt'),
                              index_col='Index', sep='\t')
            roi_labels = roi.Label.values

        elif self.atlas_name == 'AAL':
            roi_labels = self.atlas['labels']

        return roi_labels

    @property
    def masker(self):
        mask = NiftiLabelsMasker(labels_img=self.atlas_path,
                                 labels=self.atlas_labels, 
                                 memory="nilearn_cache",
                                 verbose=1,
                                 standardize='zscore_sample',
                                 detrend=True,
                                 resampling_target='labels'
                                 )    
        return mask
    
