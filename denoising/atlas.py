from pathlib import Path
import os
from typing import Optional, Union, Dict, Any, List
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_aal, fetch_atlas_schaefer_2018
import requests
from urllib.parse import urlencode


class Atlas:
    """
    Class to handle atlas file

    Attributes
    ----------
    atlas_name
        Returns atlas name
    atlas_path
        Loads atlas and returns path to atlas file
    atlas_labels
        Returns ROI labels
    masker
        Returns time series extractor instance
    """

    def __init__(self, atlas_name: str, mean_mask: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        atlas_name: str
            One of ['HCPex', 'Schaefer200', 'AAL', 'Brainnetome']
        mean_mask: str, optional
            Path to mean mask image
        
        Raise
        -----
        NotImplementedError
            If unknown atlas name is provided
        """

        if atlas_name not in ['HCPex', 'Schaefer200', 'AAL', 'Brainnetome']:
            raise NotImplementedError(
                'Available atlases: HCPex, Schaefer200, AAL, Brainnetome')

        self.atlas_name: str = atlas_name
        self.atlas_labels_path: Path = Path('../atlas/')
        self.mask: Optional[str] = mean_mask

    @property
    def atlas_path(self) -> str:
        """Return path to atlas file."""
        if self.atlas_name in ['HCPex', 'Brainnetome']:
            return self._load_atlas()

        elif self.atlas_name == 'AAL':
            self.atlas = fetch_atlas_aal(data_dir=self.atlas_labels_path)
            return self.atlas['maps']

        elif self.atlas_name == 'Schaefer200':
            self.atlas = fetch_atlas_schaefer_2018(
                n_rois=200, data_dir=self.atlas_labels_path)
            return self.atlas['maps']

    def _load_atlas(self) -> str:
        """
        Loads atlas file from Yandex Disk

        Returns
        -------
        str
            Path to loaded file
        """
        base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'

        if self.atlas_name == 'HCPex':
            public_key = 'https://disk.yandex.ru/d/AHkSrMBH8wYGHA'
            fname = os.path.join('../atlas', 'HCPex.nii.gz')

        elif self.atlas_name == 'Brainnetome':
            public_key = 'https://disk.yandex.ru/d/Fo2j_VPpS7xWiA'
            fname = os.path.join('../atlas', 'BN_Atlas_246_2mm.nii.gz')

        if os.path.exists(fname):
            return fname

        # Получаем загрузочную ссылку
        final_url = base_url + urlencode(dict(public_key=public_key))
        response = requests.get(final_url)
        download_url = response.json()['href']

        # Загружаем файл и сохраняем его
        download_response = requests.get(download_url)
        with open(fname, 'wb') as f:
            f.write(download_response.content)

        return os.path.abspath(fname)

    @property
    def atlas_labels(self) -> pd.DataFrame:
        """Return atlas labels as pandas DataFrame."""
        # Ensure atlas is loaded for AAL and Schaefer200
        if self.atlas_name in ['AAL', 'Schaefer200'] and not hasattr(self, 'atlas'):
            # Trigger atlas_path to load the atlas
            _ = self.atlas_path
        
        if self.atlas_name == 'HCPex':
            roi = pd.read_excel(os.path.join(self.atlas_labels_path, 'HCPex_sorted.xlsx'),
                                index_col='HCPex_ID')
            #roi_labels = roi.index.values
            #roi_labels = roi.sort_values(by='HCPex_ID').index.values
            #roi_labels = roi.drop([401, 365, 398], axis=0).index.values
            #roi_labels = roi.drop([396, 365, 401, 372, 405], axis=0).index.values
            roi_labels = roi.sort_index()
            #roi_labels = roi.drop([401, 372, 405], axis=0).Short_label.values

        elif self.atlas_name == 'Schaefer200':
            roi_labels = self.atlas['labels']

        elif self.atlas_name == 'AAL':
            roi_labels = self.atlas['labels']

        elif self.atlas_name == 'Brainnetome':
            roi = pd.read_csv(os.path.join(self.atlas_labels_path, 'BN_Atlas_246_LUT.txt'),
                              index_col=0, sep=' ')
            roi_labels = roi.Unknown.values

        return roi_labels

    @property
    def masker(self) -> NiftiLabelsMasker:
        """Return NiftiLabelsMasker instance for time series extraction."""
        mask = NiftiLabelsMasker(labels_img=self.atlas_path,
                                 labels=self.atlas_labels,
                                 #mask_img=self.mask,
                                 memory="nilearn_cache",
                                 verbose=1,
                                 standardize=False, #'zscore_sample',
                                 detrend=True,
                                 resampling_target='data', #'labels'
                                 n_jobs=-1 # fix
                                 )
        return mask
