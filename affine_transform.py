# cd /home/tm/projects/OpenCloseProject
# nohup /home/tm/projects/OpenCloseProject/nilearn_env/bin/python script.py &

from nilearn.image import resample_to_img
from denoising.dataset import Dataset
import nibabel as nib


derivatives_path = '/home/tm/OpenCloseProject/derivatives/' 
data = Dataset(derivatives_path, 2, 'rest')

for sub in data.sub_labels[1:]:
    ffiles = data.get_func_files(sub=sub)
    mfiles = data.get_mask_files(sub=sub)

    ref = ffiles[0]

    new_img = resample_to_img(ffiles[1], ref, 'nearest')
    new_mask = resample_to_img(mfiles[1], ref, 'nearest')

    path_to_save = f'{derivatives_path}sub-{sub}/func'
    fpath = f'{path_to_save}/sub-{sub}_task-rest_run-2_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    mpath = f'{path_to_save}/sub-{sub}_task-rest_run-2_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'

    nib.save(new_mask, mpath)
    nib.save(new_img, fpath)
