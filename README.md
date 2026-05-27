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

## Reproduction
- scripts/preprocess_for_continuation_sampling.py and scripts/preprocess_for_surprisal_studies.py used to turn stimuli files into inputs for lm-zoo
- scripts/run-lm-zoo-surprisals.sh and scripts/run-lm-zoo-get-predictions to run lm-zoo 
- scripts/process_lm-zoo-outputs.py to process surprisal outputs into readable CSV files
- Surprisal studies: the processed lm zoo outputs are analysed in notebooks notebooks/pilot_object_surprisals.ipynb and notebooks/subject_gap_surprisals.ipynb
- Continuations are generated and analysed in notebooks/subject_next_token_sampling_analysis.ipynb, notebooks/subject_generating_continuations.ipynb, 
  notebooks/generated_continuation_analysis.ipynb, notebooks/generating_continuations.ipynb, notebooks/next_token_sampling_analysis.ipynb

Note: grnn and ngram weights and vocabs extracted from lm-zoo were too large to commit to github. All results should be reproductible with jupyter notebooks and the existing files in data/model_outputs. 
Some python code was also extracted from lm-zoo's docker container for GRNN and that code is included in models/grnn.