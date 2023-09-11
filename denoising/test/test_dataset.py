import pytest
from denoising.denoise import Denoising
from denoising.atlas import Atlas
from denoising.dataset import Dataset
from denoising.helpers import *
import sys
import os
sys.path.extend([os.path.abspath('..')])

def test_dataset():
    derivatives_path = r"C:\Users\user\Desktop\open_close_001\derivatives"
    sub = ['001', '002']
    data = Dataset(sub, derivatives_path, runs=2, task='rest')