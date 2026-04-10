# perceptron-gik
Glove-based Invisible Keyboard (GIK) is a pair of gloves connected with sensors to detect finger movements via the IMU sensors on each finger, while detecting key presses with a force sensor on each finger tip. The data is collected via SPI protocol and being packeted before sending out to the inference device via Bluetooth Low Energy module (BLE) from the Arduino Nano BLE Sense Rev 2.

This project is conducted at **Imperial College London**, part of the AML Laboratory and AML Devices module where students build an hardware device that utilises ML techniques to achieve goals and tasks. 

**Project Members: Jun, Hazel, Souparna**

### Software Requirements and Setup

- Arduino IDE to write and compile C code to the Arduino Nano Microcontroller
- BMI160 IMU Sensors are used due to its small size that fits onto the fingers. Due to the availability of High-Speed SPI on Nano, we configured BMI160 to work with SPI, and we managed to make all 10 IMUs to work on a common SPI bus. Please refer to the **GIK_Nano.ino** file in the repo for the corresponding code.



