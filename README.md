# Camera Calibration

This is a camera calibration tools utilising OpenCV 3 using Python.

### Requirements

* OpenCV 3 (install via Anaconda/PIP/Brew)
* Checkerboard with known inner row/column size and real square dimension in mm
* Calibration video with different poses

### How it works?

Camera calibration is an essential step for lot of multiview geometries within computer vision. Ones can run any algorithm without doing calibration (assuming ideal *Intrinsic* camera matrix) and probably your solution will work adequately. However,
once you want to estimate the object pose relative to the camera, then *Extrinsic* camera matrix must be obtained and calibration steps must be performed.

Few great articles on camera calibration issues:

1. [Decomposing Extrinsic/Intrinsic camera matrices](http://ksimek.github.io/2012/08/14/decompose/)
2. [Uni Freiburg Slides](chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/http://ais.informatik.uni-freiburg.de/teaching/ws10/robotics2/pdfs/rob2-10-camera-calibration.pdf)
3. [OpenCV 3 Tutorial](https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html)

### Contribution

Perhaps anyone can contribute by adding different patterns for the checkerboards (circular board)?