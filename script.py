from denoising.denoise import Denoising
from denoising.atlas import Atlas
from denoising.dataset import Dataset
from denoising.helpers import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# paste derivatives path
derivatives_path = '/data/Projects/TestRetest_NYU/bids/derivatives'
# #'/data/Projects/TestRetest_NYU/bids/derivatives'
#'/arch/OpenCloseBeijin/INDI_Lite_NIFTI/derivatives/' # 
# '/arch/OpenCloseProject/derivatives/' 

# enter number of runs
runs = 3
sessions = 1
task = 'rest'
tr = 2.5

# enter atlas name
#atlases = ['AAL', 'Brainnetome', 'Schaefer200', 'HCPex']
atlas_name = 'HCPex'

folder = '/data/Projects/TestRetest_NYU/TRT_outputs'

#for atlas_name in atlases:

data = Dataset(derivatives_path=derivatives_path, 
               sessions=sessions,
               TR=tr,
               runs=runs,
               task=task)

atlas = Atlas(atlas_name=atlas_name, 
              mean_mask='/home/tm/projects/OpenCloseProject/notebooks/mean_mask_trt_03.nii.gz')

for i in range(1, 7):
    denoise = Denoising(data, atlas, 
                        # enter denoising options here
                        strategy=i, 
                        use_GSR=False, 
                        use_cosine=True, 
                        smoothing=None) 

    ts = denoise.denoise(save_outputs=True, folder=folder)
    
# cd /home/tm/projects/OpenCloseProject
# nohup /home/tm/projects/OpenCloseProject/nilearn_env/bin/python script.py &