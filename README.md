<p align="center">
    <img src="https://see.fontimg.com/api/renderfont4/KpAp/eyJyIjoiZnMiLCJoIjozNywidyI6MTAwMCwiZnMiOjM3LCJmZ2MiOiIjM0JFQkVGIiwiYmdjIjoiI0ZGRkZGRiIsInQiOjF9/QXV0b1N1cmNlSUQtRmVhdHVyZUV4dHJhY3Rvcg/kg-second-chances-sketch.png" width="120%">
</p>

<!-- Uncomment the following lines when you have a DOI or ASCL code
[![DOI](https://zenodo.org/badge/440851447.svg)](https://zenodo.org/badge/latestdoi/440851447)
<a href="https://ascl.net/2203.014"><img src="https://img.shields.io/badge/ascl-2203.014-blue.svg?colorB=262255" alt="ascl:2203.014" /></a>
-->


<p float="left">
  <img src="https://github.com/FiorenSt/AutoSourceID-FeatureExtractor/blob/main/Plots/Crowded_image_page-0001.jpg" width="23%" />
  <img src="https://github.com/FiorenSt/AutoSourceID-FeatureExtractor/blob/main/Plots/Crowded_image_2_page-0001.jpg" width="23%" /> 
  <img src="https://github.com/FiorenSt/AutoSourceID-FeatureExtractor/blob/main/Plots/Crowded_image_4_page-0001.jpg" width="23%" />
  <img src="https://github.com/FiorenSt/AutoSourceID-FeatureExtractor/blob/main/Plots/Crowded_image_3_page-0001.jpg" width="23%" /> 

</p>

## Overview

AutoSourceID-FeatureExtractor (ASID-FE) is a machine learning algorithm designed for optical images analysis. It uses single-band cutouts of 32x32 pixels around localized sources (with [ASID-L](https://github.com/FiorenSt/AutoSourceID-Light)) to estimate flux, sub-pixel centre coordinates, and their uncertainties.

#### Features
Uses a Two-Step Mean Variance Estimator (TS-MVE) approach to first estimate the features and then their uncertainties.
Does not require additional information such as Point Spread Function (PSF).
Trained on synthetic images from the MeerLICHT telescope.
Can predict more accurate features compared to similar codes like SourceExtractor.
The two-step method can estimate well-calibrated uncertainties.

## Table of Contents 
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Credits](#credits)


## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/FiorenSt/AutoSourceID-FeatureExtractor.git
   ```

# Dependencies:

* Python 3 (or superior)
* TensorFlow 2 
* Scikit-Image 0.18.1
* Numpy 1.20.3
* Astropy 4.2.1

This combination of package versions works on most Linux and Windows computers, however other package versions may also work.
If the problem persist, raise an issue and we will help you solve it.



## Usage

The use of the pre-trained ASID-FE is straight forward: 

```
python main.py
```

It loads a .fits image from the Data folder and the pre-trained model, and it outputs a catalog of features: x, y, flux, err_x, err_y, err_flux.

## License

Copyright 2023 Fiorenzo Stoppa

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.



## Credits

Credit goes to all the authors of the paper: 

**_AutoSourceID-FeatureExtractor. Optical images analysis using a Two-Step MVE Network for feature estimation and uncertainty characterization_**



