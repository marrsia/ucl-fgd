# Modelling Long-Distance Filler-Gap Dependencies

Code and data accompanying the report *Modelling of Long-Distance Filler-Gap Dependencies*.

## Repository Structure

```
├── data/
│   ├── stimuli/           # Experimental stimuli sentences
│   ├── surprisal/         # Per-token surprisal outputs from models
│   ├── predictions/       # Next-token sampling outputs and probability distributions
│   ├── wilcox_suites/     # Test suites from Wilcox et al. (2022)
│   └── mock_data/         # Mock data used for testing
├── models/                # Model weights extracted from lm-zoo (not committed, see below)
├── scripts/               # Bash and Python scripts for running models
├── utils/                 # Helper functions for text processing, data analysis, and models
└── notebooks/             # Jupyter notebooks for analysis, plot generation, and testing
```

## Models

Three models are used in this study: an n-gram model (SRILM), the Gulordava RNN (GRNN), and GPT-2. The n-gram and GRNN models are accessed via [lm-zoo](https://cpllab.github.io/lm-zoo/); GPT-2 is accessed via lm-zoo and the Hugging Face `transformers` library. Model weights for GRNN and the n-gram model were extracted from the lm-zoo Docker container and are not included in this repository due to file size.

## Dependencies

- Python 3.x
- [lm-zoo](https://cpllab.github.io/lm-zoo/)
- [Hugging Face transformers](https://huggingface.co/docs/transformers)
- spaCy
- See `requirements.txt` for full list