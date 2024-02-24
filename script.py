from denoising.denoise import Denoising
from denoising.atlas import Atlas
from denoising.dataset import Dataset
from denoising.helpers import *

from pathlib import Path

import os
import sys
sys.path.extend([os.path.abspath('/home/tm/projects/OpenCloseProject')])

# paste derivatives path
derivatives_path = '/arch/OpenCloseProject/derivatives/' #'/arch/OpenCloseBeijin/INDI_Lite_NIFTI/derivatives/'

# enter number of runs
runs = 2

# enter task name
task = 'rest'
# enter atlas name
#atlases = ['AAL', 'Brainnetome', 'Schaefer200', 'HCPex']
atlas_name = 'Brainnetome'
folder = '/home/tm/projects/OpenCloseProject/notebooks/outputs'

#for atlas_name in atlases:

data = Dataset(derivatives_path=derivatives_path, 
            runs=runs,
            task=task)

atlas = Atlas(atlas_name=atlas_name)

denoise = Denoising(data, atlas, 
                    # enter denoising options here
                    strategy=6, 
                    use_GSR=False, 
                    use_cosine=True, 
                    smoothing=None) 

#sub_labels= ['001', '002']
ts = denoise.denoise(save_outputs=True, folder=folder)

#nohup /home/tm/projects/OpenCloseProject/nilearn_env/bin/python /home/tm/projects/OpenCloseProject/script.py &