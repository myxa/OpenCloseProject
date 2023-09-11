from pathlib import Path
import os
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_aal
import requests
from urllib.parse import urlencode


class Atlas:
    def __init__(self, atlas_name):

        if atlas_name not in ['HCPex', 'Schaefer200', 'AAL']:
            raise NotImplementedError('Available atlases: HCPex, Schaefer200, AAL')

        self.atlas_name = atlas_name
        self.atlas_labels_path = Path('../atlas/')

    @property
    def atlas_path(self):
        if self.atlas_name != 'AAL':
            return self._load_atlas()
        else:
            self.atlas = fetch_atlas_aal(data_dir=self.atlas_labels_path)
            return self.atlas['maps']


    def _load_atlas(self):
        base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'

        if self.atlas_name == 'HCPex':
            public_key = 'https://disk.yandex.ru/d/mOmRumssnvS3Tw'
            fname = os.path.join('../atlas','HCPex.nii')
        elif self.atlas_name == 'Schaefer200':
            public_key = 'https://disk.yandex.ru/d/GGplzqDal5kYZg'
            fname = os.path.join('../atlas', 'Schaefer_7N_200.nii.gz')

        # Получаем загрузочную ссылку
        final_url = base_url + urlencode(dict(public_key=public_key))
        response = requests.get(final_url)
        download_url = response.json()['href']

        # Загружаем файл и сохраняем его
        download_response = requests.get(download_url)
        with open(fname, 'wb') as f:   # Здесь укажите нужный путь к файлу
            f.write(download_response.content)
        
        return os.path.abspath(fname)


    @property
    def atlas_labels(self): 

        if self.atlas_name == 'HCPex':
            roi = pd.read_excel(os.path.join(self.atlas_labels_path, 'HCPex_sorted_networks_names.xlsx'), 
                                index_col='Index')

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
    
