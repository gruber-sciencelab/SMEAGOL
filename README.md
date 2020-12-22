# SMEAGOL (Sequence Motif Enrichment And Genome annOtation Library)

Smeagol is a library to identify and visualize enrichment (or depletion) of motifs in DNA/RNA sequences.

## Setup

Install requirements (TBA)

Clone this git repo.


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

- smeagol.utils: functions to analyze PWMs and PPMs
- smeagol.visualization: functions to generate plots
- smeagol.models: tensorflow encoding of PWMs 
- smeagol.encoding: functions to encode DNA sequences
- smeagol.fastaio: functions to read and write FASTA files
- smeagol.inference: functions to score binding sites 
- smeagol.enrich: functions to calculate binding site enrichment


## Tutorials

See the [vignette](vignette_1.ipynb) for an example workflow using SMEAGOL.
