## Header
- Title: Bag of Visual Words vs YOLO: a VOC2007 comparison of two generation of object recognition approaches
- Date: 02/11/2019
- Author: Lucas dos S. Althoff
- E-mail: ls.althoff@gmail.com 

The both approaches are implemented separately and each have its own installation and execution instruction

## Implementation of BoVW

The stages of BoVW used is based on [BoVW](https://github.com/bikz05/bag-of-words) and 
[BoVW2] (https://github.com/kushalvyas/Bag-of-Visual-Words-Python)
For detail about VOC 2007 toolkit and data handling in python [VOC07Python] 
(https://github.com/mprat/pascal-voc-python) 


## Implementation of YOLO
This project is based on [darkflow](https://github.com/thtrieu/darkflow)
and [darknet](https://github.com/pjreddie/darknet).

For details about YOLO and YOLOv2 please refer to their [project page](https://pjreddie.com/darknet/yolo/) 
and the [paper](https://arxiv.org/abs/1612.08242):
YOLO9000: Better, Faster, Stronger by Joseph Redmon and Ali Farhadi.

**NOTE:**
This is still an experimental project. 

### Installation and demo
1. Clone this repository
    ```bash
    git clone https://github.com/thtrieu/darkflow
    ```

2.2.	install pip using anaconda prompt
    '''conda install -c anaconda pip'''

3. Download the trained model [yolo-voc.weights.h5](https://drive.google.com/open?id=0B4pXCfnYmG1WUUdtRHNnLWdaMEU) 
and set the model path in `demo.py`
4. Run demo `python demo.py`. 

## Dependencies

Python 3.5.2, OpenCV 3.2.0, numpy, pandas, os, matplotlib, sklearn, bs4, argparse, Glob 

# To use

## Installation
- VOC toolkit
        1. Clone the repo
        2. run `python setup.py install` or `python install develop` depending on what you will be doing

- OpenCV library

$ conda install --channel https://conda.anaconda.org/menpo opencv3
$ conda install -c anaconda numpy