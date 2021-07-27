#Installation

##   1. System Setup

This section covers the system setup of AI Flower.

## 1.1. Connect devices

Before setting up the environment, please make sure AI Flower is plug in to the socket. If you are re-assembling AI Flower, please make sure that Jetson Nano is connected correctly with the peripherals. 

## 1.2. Setup Environment 

The project is mainly implemented in Python3. Python3 should be pre-installed on Jetson Nano; if not, please make sure to install it by “sudo apt-get python3”. OpenCV is also required, which is also pre-installed. 

Then, we have to install Pytorch. A reference of implementation could be found here. Next, install torchvision by:
```
sudo apt-get install libjpeg-dev sudo apt-get install zlib1g-dev sudo apt-get install libpng-dev sudo apt-get install python3-matplotlib sudo pip install Pillow sudo pip install torchvision
```
Then, install numpy:
```
sudo apt-get install python3-numpy
```
and python image library:
```
sudo apt-get install python3-pil
```
After that, we have to install the pretrained networks used for transfer learning. Again, installation can refer to this GitHub repository:
```
git clone https://github.com/Cadene/pretrained-models.pytorch.git
cd pretrained-models.pytorch
python3 setup.py install
```
Then, we have to install Dlib face detector. To do so, first we have to install cmake and pip3:
```
sudo apt-get install git cmake
sudo apt-get install python3-pip
sudo pip3 install -U pip testresources setuptools
```
Then, install Dlib from source:
```
wget http://dlib.net/files/dlib-19.17.tar.bz2
tar xvf dlib-19.17.tar.bz2
cd dlib-19.17/
mkdir build
cd build
cmake ..
cmake --build . --config Release
sudo make install
sudo ldconfig
cd dlib-19.17
python setup.py install
pip3 install face_recognition
```
Install argparse:
```
 pip install argparse
```
Download the GitHub repository and install adafruit_servoKit accordingly:
```
./installServoKit.sh
./installGamePad.sh
python3 servoPlay.py
```
Follow this tutorial to install Adafruit_blinka. Module board and busio are included in it.

Install module serial by:

python -m pip install pyserial

##1.3. Running program

After making sure all the libraries are installed, locate “integrated.py”. Run:

python3 integrated.py

The program will now start working. Type “Ctrl” + “C” if you want to exit it at any time.


