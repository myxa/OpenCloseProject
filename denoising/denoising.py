import os

import numpy as np
import pandas as pd
from nilearn.interfaces.fmriprep import load_confounds
from tqdm.notebook import tqdm


class Denoising:

    def __init__(self, dataset, atlas,
                 strategy, n_compcor=None,
                 use_GSR=False, use_cosine=True,
                 smoothing=None):
        """ 
        
        """
        self.dataset = dataset
        self.atlas = atlas
        self.masker = atlas.masker

        self.int_strategy = strategy
        self.use_cosine = use_cosine
        self.use_GSR = use_GSR
        self.smoothing = smoothing
        self.n_compcor = n_compcor    
    
    @property
    def strategy(self): 
        # 24P
        strategy_1 = {'strategy': ['motion'],
                      'motion'  : 'full',
                      }  
        # compcor 10, 12p
        strategy_2 = {'strategy' : ['motion', 'compcor', 'high_pass'],
                      'motion'   : 'derivatives',
                      'compcor'  : 'anat_combined',
                      'n_compcor': 10}
        # compcor50, 12p
        strategy_3 = {'strategy' : ['motion', 'compcor', 'high_pass'],
                      'motion'   : 'derivatives',
                      'compcor'  : 'anat_combined',
                      'n_compcor': 'all'} 
        # compcor 10, 24p
        strategy_4 = {'strategy' : ['motion', 'compcor', 'high_pass'],
                      'motion'   : 'full',
                      'compcor'  : 'anat_combined',
                      'n_compcor': 10}
        # compcor50, 24p
        strategy_5 = {'strategy' : ['motion', 'compcor', 'high_pass'],
                      'motion'   : 'full',
                      'compcor'  : 'anat_combined',
                      'n_compcor': 'all'}

        strategy = [strategy_1, strategy_2, strategy_3, strategy_4, strategy_5][self.int_strategy-1]

        if self.use_GSR:
            strategy['strategy'].append('global_signal')
            strategy['global_signal'] = 'full' # 4p


        if self.use_cosine is False:
            self.masker.set_params(high_pass=0.008,
                                         low_pass=0.09, 
                                         t_r=2.5)
            
        if self.smoothing is not None:
            self.masker.set_params(smoothing_fwhm=self.smoothing)
            
        return strategy

    
    def denoise(self, sub=None):
        """ 
        
        """
        if sub is None:
            sub = self.dataset.sub_labels

        denoised = []
        for s in tqdm(sub):
            denoised.append(
                self.denoise_one_sub(sub=s))
        
        return denoised
    

    def denoise_one_sub(self, sub):

        imgs = self.dataset.get_func_files(sub=sub)
        assert len(imgs) == self.dataset.runs

        confounds, mask = load_confounds(imgs, **self.strategy)
        denoised_ts = []

        for i in range(self.dataset.runs):

            if self.use_cosine is False:
                confounds[i].loc[:, ~confounds[i].columns.str.startswith('cosine')]

            d = self.masker.fit_transform(imgs[i], confounds=confounds[i])
            d = np.squeeze(d)
            denoised_ts.append(d)
            _ = self.save_outputs(d, sub, run=i)

        return denoised_ts


    def save_outputs(self, outputs, sub, run):

        path_to_save = os.path.join(self.dataset.derivatives, f'sub-{sub}', 
                                    'time-series', self.atlas.atlas_name)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        name = f'sub-{sub}_task-{self.dataset.task}_run-{run+1}_time-series_{self.atlas.atlas_name}_strategy-{self.int_strategy}.csv'

        df = pd.DataFrame(outputs, columns=self.atlas.atlas_labels)
        df.to_csv(os.path.join(path_to_save, name), index=False)

        return df

