<img src="./logo.png" width="90" align="right" />

# Speaker Recognition
This repository implements a speaker recognition model utilizing Mel-Frequency Cepstral Coefficients (MFCCs) and machine learning techniques. It aims to identify a specific target speaker (labeled as "1") amidst audio recordings containing the target speaker and other individuals (labeled as "0").

## Project Structure:

*   samples: Training voice files named 0_voice and 1_voice based on speaker labels.
*   tests: Testing voice files mirroring the structure of samples.
*   app.ipynb: Jupyter Notebook for model training, evaluation, and analysis.
*   generateDataSet.py: Python script extracting MFCCs and generating datasets    (samples_dataset and tests_dataset).
*   requirements.txt: Required dependencies for project execution.


## Features

*   Data organization: Clear directory structure and naming conventions for   efficient access and management.
*   MFCC extraction: Extraction of MFCCs to capture speaker-specific vocal    characteristics.
*   Jupyter Notebook workflow: Interactive environment for model development and experimentation.
*   Machine learning model: Implemented in app.ipynb for speaker identification.

## Install
Install the required dependencies:
```sh
pip install -r requirements.txt
```

## Getting Started:

*   Install dependencies listed in requirements.txt.
*   Run generateDataSet.py to extract MFCCs and generate datasets.
*   Open app.ipynb in Jupyter Notebook for training, evaluation, and analysis.

