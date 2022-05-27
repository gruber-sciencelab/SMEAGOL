## Introduction

SMEAGOL is a user-friendly Python library to identify and visualize enrichment (or depletion) of sequence motifs in nucleic acid sequences.

SMEAGOL can load nucleic acid sequences as well as sequence motifs that represent the binding preferences of nucleic-acid binding proteins. These motifs can be analyzed and processed, and converted into Position Weight Matrices (PWMs). SMEAGOL further allows scanning of nucleic acid sequences with these motifs and scores each position of the sequence according to its match to the supplied motifs. It offers a statistical test for whether a given motif is significantly enriched or depleted in a sequence relative to a background model. Finally, SMEAGOL can load sequence variants and calculate their impact on the motif match score in their surrounding sequence.

The following sub-modules are included in this library.

- smeagol.aggregate: Aggregate results across different genomes.
- smeagol.encode: Encoding nucleic acid sequences into numeric form for PWM scanning.
- smeagol.enrich: Calculate enrichment or depletion of PWM matches in a sequence.
- smeagol.io: Read and write data (motifs and sequences).
- smeagol.matrices: Analyze motifs in the form of PFMs, PPMs and PWMs.
- smeagol.models: Encoding of PWMs into a model to scan sequences.
- smeagol.scan: Scan nucleic acid sequences with PWMs and score putative binding sites.
- smeagol.utils: Miscellaneous functions used by other modules.
- smeagol.variant: Predict the effects of sequence variants on the PWM match score.
- smeagol.visualize: Generating visualizations.

The individual functions in each sub-module are documented below.
