import numpy as np
import os
from denoising.denoise import Denoising
from denoising.atlas import Atlas

atlas = Atlas('Schaefer200')
#atlas.masker.set_params({'smoothing_fwhm': 4})

derivatives_path = '/data/Projects/ABIDE/autism/derivatives/ABIDE_pcp/cpac/filt_global'
subs = np.array([f'{i}' for i in os.listdir(derivatives_path) if 'preproc' in i])

for i in range(len(subs)):
    a = atlas.masker.fit_transform(f'{derivatives_path}/{subs[i]}')
    
    np.save(f'/data/Projects/ABIDE/autism/derivatives/ABIDE_pcp/cpac/sch/{subs[i][:-7]}.npy', a)

print('done processing')
