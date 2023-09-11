import pytest
from denoising.atlas import Atlas
import os
import sys
sys.path.extend([os.path.abspath('..')])

print(__name__)

def test_atlas_HCPex():
    atlas = Atlas(atlas_name='HCPex')
    assert len(atlas.atlas_labels) == 426
    assert atlas.atlas_path == os.path.join('../atlas','HCPex.nii')

def test_atlas_Schaefer200():
    atlas = Atlas(atlas_name='Schaefer200')
    assert len(atlas.atlas_labels) == 200
    assert atlas.atlas_path == os.path.join('../atlas','Schaefer_7N_200.nii.gz')
    
