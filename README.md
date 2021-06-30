# SMEAGOL (Sequence Motif Enrichment And Genome annOtation Library)

Smeagol is a library to identify and visualize enrichment (or depletion) of motifs in DNA/RNA sequences.

## Setup

SMEAGOL is compatible with Python 3.7.

### 1. Clone this git repository
```
git clone https://github.com/gruber-sciencelab/SMEAGOL
```

### 2. Install pip dependencies
```
cd SMEAGOL && pip install -r requirements.txt
```

### 3. Install SMEAGOL
```
pip install .
```

## Usage

In your python script / notebook, use:
```
import sys
sys.path.append('SMEAGOL') <--- supply the path to the cloned git repo here
```
You can then import modules or functions from Smeagol. For example:
```
import smeagol.visualization
```

## Modules

Smeagol contains the following modules:

- smeagol.utils: functions to analyze PPMs and PWMs
- smeagol.io: functions to read and write data
- smeagol.models: tensorflow encoding of PWMs 
- smeagol.encode: functions to encode DNA sequences
- smeagol.scan: functions to score binding sites 
- smeagol.enrich: functions to calculate binding site enrichment
- smeagol.visualize: functions to generate plots



## Tutorials

See the [vignette](vignette_1.ipynb) for an example workflow using SMEAGOL.
