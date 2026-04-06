# OpenClose Denoising Project

fMRI denoising pipeline for neuroimaging data analysis.

## Overview

This project provides tools for denoising fMRI data using various strategies including:
- 24 parameter motion regression
- aCompCor + 12 parameter regression
- aCompCor50 + 12 parameter regression
- Global signal regression options
- Multiple atlas support (HCPex, Schaefer200, AAL, Brainnetome)

## Features

- **Multiple denoising strategies**: Choose from 5 different denoising approaches
- **Atlas support**: Work with popular brain atlases
- **BIDS compatibility**: Built on pybids for standardized data organization
- **Performance optimized**: Parallel processing and caching with nilearn
- **Modern Python**: Type hints, configuration management, and modern tooling

## Installation

### Prerequisites

- Python 3.11 or 3.12
- pip (Python package manager)

### Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd OpenCloseProject
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Install optional development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Dependencies

Core dependencies are automatically installed with the package:
- numpy >=1.24.0
- pandas >=2.0.0
- scipy >=1.10.0
- nilearn >=0.10.0
- nibabel >=5.0.0
- pybids >=0.16.0
- matplotlib >=3.7.0
- seaborn >=0.12.0
- scikit-learn >=1.3.0
- tqdm >=4.65.0
- requests >=2.28.0
- openpyxl >=3.1.0 (for Excel file support)

## Usage

### Basic Example

```python
from denoising.atlas import Atlas
from denoising.dataset import Dataset
from denoising.denoise import Denoising

# Initialize atlas
atlas = Atlas('Schaefer200')

# Initialize dataset
dataset = Dataset(
    derivatives_path='/path/to/derivatives',
    TR=2.0,
    sessions=1,
    runs=2,
    task='rest'
)

# Create denoising pipeline
denoise = Denoising(
    dataset=dataset,
    atlas=atlas,
    strategy=1,  # 24 parameter regression
    use_GSR=False,
    use_cosine=True
)

# Process a subject
# denoise.denoise('sub-01')
```

### Configuration

The project supports configuration through:
1. Environment variables:
   ```bash
   export DERIVATIVES_PATH=/path/to/derivatives
   export OUTPUT_PATH=./output
   export N_JOBS=-1  # Use all available cores
   ```

2. Configuration file (`config.yaml`):
   ```yaml
   data:
     derivatives_path: /path/to/derivatives
     output_path: ./output
     atlas_path: ./atlas
   
   processing:
     n_jobs: -1
     memory: nilearn_cache
     verbose: 1
   ```

3. Programmatic configuration:
   ```python
   from denoising.config import get_config
   config = get_config()
   config.set('data', 'derivatives_path', '/custom/path')
   ```

### Command Line Interface

After installation, you can use the command-line interface:

```bash
openclose-denoise --help
```

## Project Structure

```
OpenCloseProject/
├── denoising/          # Main package
│   ├── __init__.py
│   ├── atlas.py       # Atlas handling
│   ├── dataset.py     # BIDS dataset management
│   ├── denoise.py     # Denoising algorithms
│   ├── config.py      # Configuration management
│   ├── connectivity.py
│   ├── coverage.py
│   ├── helpers.py
│   ├── metrics.py
│   └── test/          # Unit tests
├── atlas/             # Atlas data files
├── notebooks/         # Example notebooks
├── tests/             # Integration tests
├── pyproject.toml     # Project configuration
├── README.md          # This file
└── run_denoise.py     # Example script
```

## Atlas Support

The project supports four atlases:

1. **HCPex**: 426 regions, requires downloading from Yandex Disk
2. **Schaefer200**: 200 regions from Schaefer 2018 atlas
3. **AAL**: Automated Anatomical Labeling atlas
4. **Brainnetome**: 246 regions, requires downloading from Yandex Disk

For HCPex and Brainnetome atlases, the files will be automatically downloaded from Yandex Disk on first use.

## Development

### Setting Up Development Environment

1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Run tests:
   ```bash
   pytest denoising/test/ -v
   ```

3. Type checking:
   ```bash
   mypy denoising/
   ```

4. Code formatting:
   ```bash
   black denoising/
   ```

5. Linting:
   ```bash
   flake8 denoising/
   ```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Notebooks

Example notebooks are available in the `notebooks/` directory:

- `example.ipynb`: Basic usage examples
- `extract_roi_ts.ipynb`: ROI time series extraction
- `fc_estimation.ipynb`: Functional connectivity estimation
- `metrics.ipynb`: Quality metrics calculation
- `to_bids.ipynb`: BIDS format conversion

## License

MIT License - see LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```
[Citation information to be added]
```

## Support

For issues and questions, please use the GitHub issue tracker or contact the maintainers.

## Acknowledgments

- Built on top of [nilearn](https://nilearn.github.io/) and [pybids](https://bids-standard.github.io/pybids/)
- Atlas data from various sources (see individual atlas files for citations)
- Contributors and users of the OpenClose project