[![facialAnalysisLiteDetection2D Homepage](https://img.shields.io/badge/facialAnalysisLiteDetection2D-develop-orange.svg)](https://github.com/davidvelascogarcia/facialAnalysisLiteDetection2D/tree/develop/programs) [![Latest Release](https://img.shields.io/github/tag/davidvelascogarcia/facialAnalysisLiteDetection2D.svg?label=Latest%20Release)](https://github.com/davidvelascogarcia/facialAnalysisLiteDetection2D/tags) [![Build Status](https://travis-ci.org/davidvelascogarcia/facialAnalysisLiteDetection2D.svg?branch=develop)](https://travis-ci.org/davidvelascogarcia/facialAnalysisLiteDetection2D)

# Facial Analysis: Lite Detector 2D (Python API)

- [Introduction](#introduction)
- [Trained Models](#trained-models)
- [Requirements](#requirements)
- [Status](#status)
- [Related projects](#related-projects)


## Introduction

`facialAnalysisLiteDetection2D` module use `tensorflow` , `cvlib` and `openCV` `python` API. The module analyze faces using pre-trained models and adds facial analysis doing prediction about some features like gender, age and emotions. Also use `YARP` to send video source pre and post-procesed. Also admits `YARP` source video like input. This module also publish detection results in `YARP` port.

Documentation available on [docs](https://davidvelascogarcia.github.io/facialAnalysisLiteDetection2D)


## Trained Models

`facialAnalysisLiteDetection2D` requires images source to detect. First run program will download `gender` pre-trained models, the rest of the models are located in [models](./models) dir, about features to detect:

1. Execute [programs/facialAnalysisLiteDetection2D.py](./programs), to start the program.
```python
python3 facialAnalysisLiteDetection2D.py
```
3. Connect video source to `facialAnalysisLiteDetection2D`.
```bash
yarp connect /videoSource /facialAnalysisLiteDetection2D/img:i
```

NOTE:

- Video results are published on `/facialAnalysisLiteDetection2D/img:o`
- Data results are published on `/facialAnalysisLiteDetection2D/data:o`

## Requirements

`facialAnalysisLiteDetection2D` requires:

* [Install OpenCV 3.4.7+](https://github.com/roboticslab-uc3m/installation-guides/blob/master/install-opencv.md)
* [Install YARP 2.3.XX+](https://github.com/roboticslab-uc3m/installation-guides/blob/master/install-yarp.md)
* [Install pip](https://github.com/roboticslab-uc3m/installation-guides/blob/master/install-pip.md)
* Install tensorflow 1.13.1+:
```bash
pip3 install tensorflow==1.13.1
```
* Install cvlib:
```bash
pip3 install cvlib
```

Tested on: `windows 10`, `ubuntu 14.04`, `ubuntu 16.04`, `ubuntu 18.04`, `lubuntu 18.04` and `raspbian`.


## Status

[![Build Status](https://travis-ci.org/davidvelascogarcia/facialAnalysisLiteDetection2D.svg?branch=develop)](https://travis-ci.org/davidvelascogarcia/facialAnalysisLiteDetection2D)

[![Issues](https://img.shields.io/github/issues/davidvelascogarcia/facialAnalysisLiteDetection2D.svg?label=Issues)](https://github.com/davidvelascogarcia/facialAnalysisLiteDetection2D/issues)

## Related projects

* [arunponnusamy: cvlib project](https://github.com/arunponnusamy/cvlib)
* [atulapra: Emotion-detection project](https://github.com/atulapra/Emotion-detection)

