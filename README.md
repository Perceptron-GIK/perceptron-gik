# perceptron-gik
Glove-based Invisible Keyboard

### Software Requirements (to allow Micropython on Arduino, we need special Arduino editor)
- MicroPython Firmware Installer https://github.com/arduino/lab-micropython-installer.git - the firmware installer is needed to install MicroPython on our Arduino board.
- Arduino Lab for Micropython https://labs.arduino.cc/en/labs/micropython - Arduino Lab for MicroPython is an editor where we can create and run MicroPython scripts on our Arduino board.

### How MicroPython works on NanoBLE Sense Rev2
- Download the MicroPython firmware onto the Nano board. Note that once this is done, you will not be able to upload .ino files onto the board anymore. 
- The structure of the .py scripts are fixed, namely boot.py and main.py, which are the two files that will only be executed offline when the board is running in standalone condition. boot.py will run once whenever Nano is powered up for the first time/reset, thus please put the initialisation and import modules in this file.
- main.py runs after boot.py and will be executed till the board is disconnected from power supply. Thus please put the main loop logic inside this file. 