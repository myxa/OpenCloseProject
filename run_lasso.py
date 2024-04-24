from denoising.helpers import *
from denoising.metrics import ICC, bss
from denoising.lasso import *
from nilearn.connectome import sym_matrix_to_vec
import pickle
import numpy as np

atlas = 'HCPex' # Schaefer200 AAL Brainnetome
strategy = 6

closed = fetch_ts('/data/Projects/OpenCloseIHB/outputs',
                  #sub=['001'], m 
                  run=1, atlas_name=atlas, strategy=strategy)

opened = fetch_ts('/data/Projects/OpenCloseIHB/outputs',
                  #sub=['001'], 
                  run=2, atlas_name=atlas, strategy=strategy)

for i in range(len(closed)):
    glassoParCorr, cvResults = graphicalLasso(closed[i].T)
    np.save(f'/home/tm/projects/OpenCloseProject/lasso_outputs/ihb/close_lasso_sub-{i}.npy', glassoParCorr)
    
    with open(f'/home/tm/projects/OpenCloseProject/cv_results/ihb/close_lasso_cv_sub-{i}.pickle', 'wb') as handle:
        pickle.dump(cvResults, handle, protocol=pickle.HIGHEST_PROTOCOL)