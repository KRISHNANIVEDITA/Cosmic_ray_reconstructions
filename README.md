# RET-CR Surface Reconstruction

This repository contains cosmic-ray reconstruction and analysis code developed for the surface stations of the Radar Echo Telescope for Cosmic Rays (RET-CR).

## Overview

RET-CR uses five surface stations equipped with scintillator panels to detect cosmic-ray air showers. The code reconstructs key shower properties from the surface-station measurements, including:

- Arrival direction from the relative signal-arrival times.
- Shower-core position using the lateral distribution of particle deposits.
- Primary cosmic-ray energy using detector simulations and the reconstructed shower profile.
- Recovery of energy deposits for events containing saturated scintillator signals.
- Utilities for event selection, detector geometry, data handling and visualisation.

The saturation analysis estimates deposits above the detector’s ADC limit by testing possible values and identifying the solution that best describes the measured lateral distribution.

The reconstructed geometry and energy provide essential inputs for the corresponding search for signals in the RET-CR in-ice radar system.

## Data and usage

The scripts operate on calibrated RET-CR scintillator data. Experimental data are not included because they are managed by the Radar Echo Telescope Collaboration. Users must provide input data in the expected format and adapt the relevant input and output paths.

The analysis uses Python and standard scientific-computing packages, including NumPy, SciPy and Matplotlib. Additional dependencies are specified within the individual scripts.

## Development status

This repository contains research code used in the RET-CR surface-station analysis. The repository is being progressively organised and documented, and its structure and interfaces may therefore change between versions.

## Related publication

K. N. Gopinath and K. Mulrey, on behalf of the Radar Echo Telescope Collaboration (2025).  
“High energy cosmic ray reconstructions with surface stations of RET-CR.”  
*Proceedings of Science*, PoS(ICRC2025)274.  
DOI: [10.22323/1.501.0274](https://doi.org/10.22323/1.501.0274)

## Author

Developed by **Krishna Nivedita Gopinath** as part of doctoral research at Radboud University and within the Radar Echo Telescope Collaboration.
