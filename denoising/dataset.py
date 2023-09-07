from pathlib import Path

import bids
import pandas as pd


class Dataset:

    def __init__(self, sub_label: list, derivatives_path: str, runs: int,
                 task: str):
        """ 
        
        """
        self.sub_labels = sub_label
        self.derivatives = Path(derivatives_path).as_posix()
        self.runs = runs # сколько 
        self.task = task

    @property
    def bids_layout(self):
        return bids.BIDSLayout(
                self.derivatives, validate=False, config=['bids','derivatives'])
    

    def get_confounds_one_subject(self, sub=None):
        conf_paths = self.bids_layout.get(subject=sub,
                                          extension='tsv',
                                          return_type='file')
        return [pd.read_csv(conf_paths[i], sep='\t') for i in range(self.runs)]
    
            
    def get_func_files(self, sub=None):
        if sub is None:
            sub = self.sub_labels
        return self.bids_layout.get(subject=sub,
                                    datatype='func', 
                                    task=self.task,
                                    desc='preproc',
                                    space='MNI152NLin2009cAsym',
                                    extension='nii.gz',
                                    return_type='file')
        