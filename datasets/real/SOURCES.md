# Real Data Sources

This directory contains small samples of real, publicly available neural
recordings used by the `ReplayStreamer` and `fMRIProcessor` to demonstrate
the signal-processing pipeline on real data rather than synthetic signals.

## EEG: PhysioNet EEG Motor Movement/Imagery Dataset

- **Files**: `eeg/S001R01.edf`, `eeg/S001R04.edf`
- **Source**: https://physionet.org/content/eegmmidb/1.0.0/
- **Subject**: S001 (baseline eyes-open run and a motor imagery run)
- **Format**: EDF, 64 channels, 160 Hz
- **License**: Open Data Commons Attribution License v1.0 (ODC-BY)
- **Citation**:
  - Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R.
    BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.
    IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.
  - Goldberger, A., et al. PhysioBank, PhysioToolkit, and PhysioNet:
    Components of a New Research Resource for Complex Physiologic Signals.
    Circulation 101(23):e215-e220, 2000.
  - DOI: 10.13026/C28G6P

The channel names in the EDF files (e.g. `Fp1.`, `F3..`) carry trailing dots
from the original recording montage. `ReplayStreamer` strips these dots and
selects the channels matching the default `EEGConfig.channels`
(`Fp1, Fp2, F3, F4, C3, C4, P3, P4`).

## fMRI: OpenNeuro ds000001 (Balloon Analog Risk-taking Task)

- **File**: `fmri/sub-01_inplaneT2.nii.gz`
- **Source**: https://openneuro.org/datasets/ds000001/
- **Subject**: sub-01, in-plane T2 anatomical volume
- **Format**: NIfTI (.nii.gz)
- **License**: CC0 (Public Domain)
- **Citation**:
  - Schonberg, T., Fox, C.R., Mumford, J.A., Congdon, E., Trepel, C.,
    Poldrack, R.A. Decreasing ventromedial prefrontal cortex activity
    during sequential risk-taking: an FMRI investigation of the balloon
    analog risk task. Frontiers in Neuroscience 6:80, 2012.

`fMRIProcessor.process_file()` loads this volume directly with `nibabel` and
reduces it to ROI-averaged time series.
