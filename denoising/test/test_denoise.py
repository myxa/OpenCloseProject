import pytest
from denoising.denoise import Denoising
from denoising.atlas import Atlas
from denoising.dataset import Dataset
from denoising.helpers import *


def test_denoising():
    atlas = Atlas('AAL')
    derivatives_path = r"C:\Users\user\Desktop\open_close_001\derivatives"
    dataset = Dataset(derivatives_path, 2, 'rest')
    
    denoise = Denoising(dataset, atlas, 
                        # enter denoising options here
                        strategy=1, 
                        use_GSR=False, 
                        use_cosine=False, 
                        smoothing=None)
    
    denoise.masker
    