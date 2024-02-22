from denoising.denoise import Denoising
from denoising.atlas import Atlas
from denoising.dataset import Dataset
import pytest 

@pytest.fixture
def dataset():
    # paste derivatives path
    derivatives_path = '/arch/OpenCloseProject/derivatives/'
    # enter number of runs
    runs = 2
    # enter task name
    task = 'rest'

    return Dataset(derivatives_path=derivatives_path,
                   runs=runs,
                   task=task)


@pytest.fixture
def atlas(atlas_name):
    return Atlas(atlas_name=atlas_name)


def test_Atlas_HCPex():
    atlas = atlas(atlas_name='HCPex')
    
    atlas.masker.labels.shape[0] 









def test_Denoising():
    denoise = Denoising(data, atlas,
                        # enter denoising options here
                        strategy=5,
                        use_GSR=False,
                        use_cosine=True,
                        smoothing=None)


def test_Dataset():
    # paste derivatives path
    derivatives_path = '/arch/OpenCloseProject/derivatives/'

    # enter number of runs
    runs = 2

    # enter task name
    task = 'rest'



    data = Dataset(derivatives_path=derivatives_path, 
                runs=runs,
                task=task)
