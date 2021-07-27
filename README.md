# AI-Flower
## Introduction

<img src="https://github.com/g7an/AI-Flower/blob/main/images/cover_page.png" width="800" />
<!-- ![](https://github.com/g7an/AI-Flower/blob/main/images/cover_page.png) -->

AI Flower is a flower shaped robot with expression recognition function. It is my Honour's project that I worked on for two semesters during my undergraduate study. It is enlightened by an [existing experimental product](https://github.com/androidthings/experiment-expression-flower) created by Google using Android Things.

In this project, I refered to the design of the existing project, and trained my own neural network for expression recognition using PyTorch. In order to have more computing power, I used Nvdia Jetson Nano instead of Raspberry Pi.

## Description

### Expression Recognition

When the programs starts to run, camera at the center of the flower starts capturing. When **Dlib Face Detector** recognizes human faces, it will crop out only the face segment in the streamed video. The face segment will be treated as the input of neural network. 

The neural network used in this project is **Inception-ResNet-v2**, using the method of transfer learning. The model is trained by dataset with more than 8000 happy and unhappy faces, with an overall accuracy of around 80\%.

### Inference time

- Average processing time for Dlib face detector is **2.20 seconds**
- Average processing time for emotion recognition model is **0.49 seconds**


### Human Computer Interaction

<br />
<img src="https://github.com/g7an/AI-Flower/blob/main/images/hci.png" width="600" />
<br />

### Technical Overview

<br />
<img src="https://github.com/g7an/AI-Flower/blob/main/images/flow.png" width="800" />
<br />

### State changing flow

<br />
<img src="https://github.com/g7an/AI-Flower/blob/main/images/led_control.png" width="800" />
<br />

### Assembly Structure

<br />
<img src="https://github.com/g7an/AI-Flower/blob/main/images/servo.png" width="800" />
<br />

### Others

More details to be found on FYP Final Presentation.pdf
