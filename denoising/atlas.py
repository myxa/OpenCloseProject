from pathlib import Path
import os
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

    def __init__(self, atlas_name):
        """
        Parameters
        ----------
        atlas_name: str
            One of ['HCPex', 'Schaefer200', 'AAL', 'Brainnetome']
        
        Raise
        -----
        NotImplementedError
            If unknown atlas name is provided
        """

        if atlas_name not in ['HCPex', 'Schaefer200', 'AAL', 'Brainnetome']:
            raise NotImplementedError(
                'Available atlases: HCPex, Schaefer200, AAL, Brainnetome')

        self.atlas_name = atlas_name
        self.atlas_labels_path = Path('../atlas/')

    @property
    def atlas_path(self):
        if self.atlas_name in ['HCPex', 'Brainnetome']:
            return self._load_atlas()

        elif self.atlas_name == 'AAL':
            self.atlas = fetch_atlas_aal(data_dir=self.atlas_labels_path)
            return self.atlas['maps']

        elif self.atlas_name == 'Schaefer200':
            self.atlas = fetch_atlas_schaefer_2018(
                n_rois=200, data_dir=self.atlas_labels_path)
            return self.atlas['maps']

    def _load_atlas(self):
        """
        Loads atlas file from Yandex Disk

        Returns
        -------
        path to loaded file
        """
        base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'

        if self.atlas_name == 'HCPex':
            public_key = 'https://disk.yandex.ru/d/AHkSrMBH8wYGHA'
            fname = os.path.join('../atlas', 'HCPex.nii.gz')

        elif self.atlas_name == 'Brainnetome':
            public_key = 'https://disk.yandex.ru/d/xQQaUOhgP7oetA'
            fname = os.path.join('../atlas', 'BN_Atlas_246_1mm.nii.gz')

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
    def atlas_labels(self):

        if self.atlas_name == 'HCPex':
            roi = pd.read_excel(os.path.join(self.atlas_labels_path, 'HCPex_sorted.xlsx'),
                                index_col='NEW_ID')
            roi_labels = roi.sort_values(by='HCPex_ID').Label.values

        elif self.atlas_name == 'Schaefer200':
            roi_labels = self.atlas['labels']

        elif self.atlas_name == 'AAL':
            roi_labels = self.atlas['labels']

        elif self.atlas_name == 'Brainnetome':
            roi = pd.read_csv('../atlas/BN_Atlas_246_LUT.txt',
                              index_col=0, sep=' ')
            roi_labels = roi.Unknown.values

        return roi_labels

    @property
    def masker(self):
        mask = NiftiLabelsMasker(labels_img=self.atlas_path,
                                 labels=self.atlas_labels,
                                 memory="nilearn_cache",
                                 verbose=1,
                                 standardize=False, #'zscore_sample',
                                 detrend=True,
                                 resampling_target='data', #'labels'
                                 n_jobs=32
                                 )
        return mask
