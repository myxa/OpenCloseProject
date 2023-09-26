from pathlib import Path

import bids
import pandas as pd


class Dataset:
    """
    Class to handle fmriprep output

    Attributes
    ----------
    bids_layout
        Returns dataset layout
    sub_labels
        Returns a list of subject labels
    
    Methods
    -------
    get_func_files(sub)
        Returns list of functional files paths

    """

    def __init__(self, derivatives_path: str, runs: int,
                 task: str):
        """ 
        Parameters
        ----------
        derivatives_path: str
            Path to derivatives directory
        runs: int
            Number of runs in one folder
        task: str
            Task name
        """

        self.derivatives = Path(derivatives_path).as_posix()
        self.runs = runs # сколько 
        self.task = task

    @property
    def bids_layout(self):
        return bids.BIDSLayout(
                self.derivatives, validate=False, config=['bids','derivatives'])
    
    @property
    def sub_labels(self):
        return self.bids_layout.get_subjects()
    

    def _get_confounds_one_subject(self, sub=None):
        conf_paths = self.bids_layout.get(subject=sub,
                                          extension='tsv',
                                          return_type='file')
        return [pd.read_csv(conf_paths[i], sep='\t') for i in range(self.runs)]
    
            
    def get_func_files(self, sub=None):
        """
        Get functional files' paths

        Parameters
        ----------
        sub: list of str, optional
            Subject labels to get files for. If None, all subjects are returned
        
        Returns
        -------
        list of str
        """

        if sub is None:
            sub = self.sub_labels
        return self.bids_layout.get(subject=sub,
                                    datatype='func', 
                                    task=self.task,
                                    desc='preproc',
                                    space='MNI152NLin2009cAsym',
                                    extension='nii.gz',
                                    return_type='file')
    
        