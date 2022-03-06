# SMEAGOL (Sequence Motif Enrichment And Genome annOtation Library)

Smeagol is a library to identify and visualize enrichment (or depletion) of motifs in DNA/RNA sequences.

## Setup

It is recommended to install SMEAGOL in a conda environment or virtualenv. SMEAGOL is compatible with Python 3.7 and higher.

If you have conda installed on your machine you can create a conda environment like this:
```
conda create --name SMEAGOL python=3.7
```

Next, you need to activate the created environment before you can install SMEAGOL into it, which can be done as follows:

```
conda activate SMEAGOL
```

### 1. Clone this git repository
```
git clone https://github.com/gruber-sciencelab/SMEAGOL && cd SMEAGOL
```

### 2. Install ghostscript if needed

Some of SMEAGOL's visualization functions require [ghostscript](https://www.ghostscript.com/). If you do not have ghostscript installed, please see the link for installation instructions or use `conda install -c conda-forge ghostscript`. (Installing ghostscript via pip leads to an error).

### 3. Install SMEAGOL along with pip dependencies
```
pip install .
```
Or, if you want to edit the code:
```
pip install -e .
```

### 4. Run tests locally (optional)
```
cd tests
pytest
```

### 5. Add SMEAGOL kernel to Ipython
```
python -m ipykernel install --user --name=SMEAGOL --display-name='Python 3.7 (SMEAGOL)'
```

## Usage

In your python script / notebook, you can import modules or functions from SMEAGOL. For example:
```
import smeagol.visualize
```
```
from smeagol.visualize import plot_background
```

## Modules

Smeagol contains the following modules:

- smeagol.matrices: functions to analyze PPMs and PWMs
- smeagol.io: functions to read and write data
- smeagol.models: tensorflow encoding of PWMs 
- smeagol.encode: functions to encode DNA sequences
- smeagol.scan: functions to score binding sites 
- smeagol.enrich: functions to calculate binding site enrichment
- smeagol.variant: functions to predict the effects of sequence variants
- smeagol.visualize: functions to generate plots



## Tutorials

See the [vignette](vignette_1.ipynb) for an example workflow using SMEAGOL.


