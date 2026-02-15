//-------------------------------------------------------------------------------------------------------------------
// This code is to make the Arduino Nano BLE Sense Rev 2 on GIK Left Hand to collect the data from all the sensors
// The sensor values are sent to the receiver via Bluetooth service with the MTU size of 153 bytes
//--------------------------------------------------------------------------------------------------------------------

//#define LEFT_HAND  // Define this as left hand 
#define RIGHT_HAND  // Define this as right hand
#include "GIK_Hand_Config.h"  // Include the hand configuration file
#include <ArduinoBLE.h>
#include "Arduino_BMI270_BMM150.h"

// For Finger IMU libraries
#include <SPI.h>
#include <Adafruit_BMP280.h>
#include <BMI160Gen.h>

BMI160GenClass BMI160_thumb;
BMI160GenClass BMI160_index;
BMI160GenClass BMI160_middle;
BMI160GenClass BMI160_ring;
BMI160GenClass BMI160_pinky;

BLEService GIK_Service(ServiceID);  // service ID
BLECharacteristic GIK_tx_Char(CharID,BLERead | BLENotify,153); // Characteristic ID with notification and MTU of 153 BYTES
Adafruit_BMP280 BMP280;

const char* Name = Hand_Name;

const int CS_THUMB = 3; // Chip Select pin for talking to thumb IMU
const int CS_INDEX = 9; // Chip Select pin for talking to index IMU
const int CS_MIDDLE = 8; // Chip Select pin for talking to middle IMU
const int CS_RING = 7; // Chip Select pin for talking to ring IMU
const int CS_PINKY = 6; // Chip Select pin for talking to pinky IMU

// Define FSR Pins

#define FSR1_PIN A0
#define FSR2_PIN A1
#define FSR3_PIN A2
#define FSR4_PIN A3
#define FSR5_PIN A6
#define THRESHOLDHIGH 300
#define THRESHOLDLOW 5

// Definition of variables for left hand

float ax_base = 0, ay_base = 0, az_base = 0; // accelerometer xyz from left base 
float gx_base = 0, gy_base = 0, gz_base = 0; // gyro xyz from left base

float ax_tb, ay_tb, az_tb, gx_tb, gy_tb, gz_tb;
int ax_thumb = 0, ay_thumb = 0, az_thumb = 0; // accelerometer xyz from left thumb 
int gx_thumb = 0, gy_thumb = 0, gz_thumb = 0; // gyro xyz from left thumb

float ax_id, ay_id, az_id, gx_id, gy_id, gz_id;
int ax_index = 0, ay_index = 0, az_index = 0; // accelerometer xyz from left index
int gx_index = 0, gy_index = 0, gz_index = 0; // gyro xyz from left index

float ax_m, ay_m, az_m, gx_m, gy_m, gz_m;
int ax_middle = 0, ay_middle = 0, az_middle = 0; // accelerometer xyz from left middle
int gx_middle = 0, gy_middle = 0, gz_middle = 0; // gyro xyz from left middle

float ax_r, ay_r, az_r, gx_r, gy_r, gz_r;
int ax_ring = 0, ay_ring = 0, az_ring = 0; // accelerometer xyz from left ring 
int gx_ring = 0, gy_ring = 0, gz_ring = 0;  // gyro xyz from left ring

float ax_p, ay_p, az_p, gx_p, gy_p, gz_p;
int ax_pinky = 0, ay_pinky = 0, az_pinky = 0; // accelerometer xyz from left pinky
int gx_pinky = 0, gy_pinky = 0, gz_pinky = 0; // gyro xyz from left pinky

// Packet layout (little-endian):
// uint32  sample_id
// float   ax_base, ay_base, az_base, gx_base, gy_base, gz_base
// for each finger: thumb, index, middle, ring, pinky
//   float ax, ay, az, gx, gy, gz
//   uint8 f which should add up to 153 bytes of MTU


uint32_t sample_id = 0; // to label the packets

void setup() {
  Serial.begin(9600);

  Serial.println("After Serial");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  if (!BLE.begin()) {
    Serial.println("Failed to initialize BLE!");
    while (1);
  }

  // Initialise SPI for finger IMUs
  SPI.begin();
  Serial.println("After SPI");

  //Set all CS pins HIGH (deselected) before initializing
  pinMode(CS_THUMB, OUTPUT);  digitalWrite(CS_THUMB, HIGH);
  pinMode(CS_INDEX, OUTPUT);  digitalWrite(CS_INDEX, HIGH);
  pinMode(CS_MIDDLE, OUTPUT); digitalWrite(CS_MIDDLE, HIGH);
  pinMode(CS_RING, OUTPUT);   digitalWrite(CS_RING, HIGH);
  pinMode(CS_PINKY, OUTPUT);  digitalWrite(CS_PINKY, HIGH);

  Serial.println("Before Init");
  
  // Initialize each BMI160 one at a time (CS pin only - library auto-detects SPI)
  BMI160_thumb.begin(CS_THUMB);
  BMI160_thumb.setGyroRange(250);
  BMI160_thumb.setAccelerometerRange(4);
  Serial.print("THUMB ID: "); Serial.println(BMI160_thumb.getDeviceID(), HEX);

  BMI160_index.begin(CS_INDEX);
  BMI160_index.setGyroRange(250);
  BMI160_index.setAccelerometerRange(4);
  Serial.print("INDEX ID: "); Serial.println(BMI160_index.getDeviceID(), HEX);

  BMI160_middle.begin(CS_MIDDLE);
  BMI160_middle.setGyroRange(250);
  BMI160_middle.setAccelerometerRange(4);
  Serial.print("MIDDLE ID: "); Serial.println(BMI160_middle.getDeviceID(), HEX);

  BMI160_ring.begin(CS_RING);
  BMI160_ring.setGyroRange(250);
  BMI160_ring.setAccelerometerRange(4);
  Serial.print("RING ID: "); Serial.println(BMI160_ring.getDeviceID(), HEX);

  BMI160_pinky.begin(CS_PINKY);
  BMI160_pinky.setGyroRange(250);
  BMI160_pinky.setAccelerometerRange(4);
  Serial.print("PINKY ID: "); Serial.println(BMI160_pinky.getDeviceID(), HEX);

  Serial.println("After Init");

  BLE.setLocalName(Name);
  BLE.setConnectionInterval(6,16); // Important: tells how often the packets should be sent (1.25ms resolution)
  // limit is 6*1.25 = 7ms and 16*1.25ms = 20ms
  BLE.setAdvertisedService(GIK_Service);

  GIK_Service.addCharacteristic(GIK_tx_Char);
  BLE.addService(GIK_Service);

  BLE.advertise();
  Serial.println("Advertising, waiting for receiver to activate...");

  // set the on board RGB as output (high turn off the RGB, low turn them on)
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);

}

void loop() {

  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  digitalWrite(LEDR, LOW);
  BLEDevice central = BLE.central();   // Wait here until receiver.py connects (havent subscribe)

  if (central) {
    Serial.print("Connected to Receiver: ");
    Serial.println(central.address());
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDB, HIGH);
    digitalWrite(LEDG, LOW);

    // Reset sample counter on each new connection
    sample_id = 0;

    while (!GIK_tx_Char.subscribed() && central.connected()) {
      delay(100); // Wait until receiver.py subscribes then only move to data sending to prevent sample index loss
    }
    
    delay(500);
    while (central.connected()) {
  
      unsigned long startTime = micros();


      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(ax_base, ay_base, az_base);
      }

      if (IMU.gyroscopeAvailable()) {
        IMU.readGyroscope(gx_base, gy_base, gz_base);
      }

      bool f_thumb = analogRead(FSR1_PIN) > THRESHOLDLOW ? 1 : 0;
      bool f_index = analogRead(FSR2_PIN) > THRESHOLDLOW ? 1 : 0;
      bool f_middle = analogRead(FSR3_PIN) > THRESHOLDLOW ? 1 : 0;
      bool f_ring = analogRead(FSR4_PIN) > THRESHOLDLOW ? 1 : 0;
      bool f_pinky = analogRead(FSR5_PIN) > THRESHOLDLOW ? 1 : 0;

      // Serial.print("thumb=");  Serial.print(analogRead(FSR1_PIN));
      // Serial.print(" index="); Serial.print(analogRead(FSR2_PIN));
      // Serial.print(" middle=");Serial.print(analogRead(FSR3_PIN));
      // Serial.print(" ring=");  Serial.print(analogRead(FSR4_PIN));
      // Serial.print(" pinky="); Serial.println(analogRead(FSR5_PIN));

      // Read each finger IMU - library handles CS internally after begin()
      digitalWrite(CS_THUMB, LOW);
      BMI160.readMotionSensor(ax_thumb, ay_thumb, az_thumb, gx_thumb, gy_thumb, gz_thumb);
      digitalWrite(CS_THUMB, HIGH);

      digitalWrite(CS_INDEX, LOW); 
      BMI160.readMotionSensor(ax_index, ay_index, az_index, gx_index, gy_index, gz_index);
      digitalWrite(CS_INDEX, HIGH);

      digitalWrite(CS_MIDDLE, LOW);
      BMI160.readMotionSensor(ax_middle, ay_middle, az_middle, gx_middle, gy_middle, gz_middle);
      digitalWrite(CS_MIDDLE, HIGH);

      digitalWrite(CS_RING, LOW); 
      BMI160.readMotionSensor(ax_ring, ay_ring, az_ring, gx_ring, gy_ring, gz_ring);
      digitalWrite(CS_RING, HIGH); 

      digitalWrite(CS_PINKY, LOW);
      BMI160.readMotionSensor(ax_pinky, ay_pinky, az_pinky, gx_pinky, gy_pinky, gz_pinky);
      digitalWrite(CS_PINKY, HIGH);

      ax_tb = convertRawAccel(ax_thumb);
      ay_tb = convertRawAccel(ay_thumb);
      az_tb = convertRawAccel(az_thumb);
      gx_tb = convertRawGyro(gx_thumb);
      gy_tb = convertRawGyro(gy_thumb);
      gz_tb = convertRawGyro(gz_thumb);

      ax_id = convertRawAccel(ax_index);
      ay_id = convertRawAccel(ay_index);
      az_id = convertRawAccel(az_index);
      gx_id = convertRawGyro(gx_index);
      gy_id = convertRawGyro(gy_index);
      gz_id = convertRawGyro(gz_index);

      ax_m = convertRawAccel(ax_middle);
      ay_m = convertRawAccel(ay_middle);
      az_m = convertRawAccel(az_middle);
      gx_m = convertRawGyro(gx_middle);
      gy_m = convertRawGyro(gy_middle);
      gz_m = convertRawGyro(gz_middle);

      ax_r = convertRawAccel(ax_ring);
      ay_r = convertRawAccel(ay_ring);
      az_r = convertRawAccel(az_ring);
      gx_r = convertRawGyro(gx_ring);
      gy_r = convertRawGyro(gy_ring);
      gz_r = convertRawGyro(gz_ring);

      ax_p = convertRawAccel(ax_pinky);
      ay_p = convertRawAccel(ay_pinky);
      az_p = convertRawAccel(az_pinky);
      gx_p = convertRawGyro(gx_pinky);
      gy_p = convertRawGyro(gy_pinky);
      gz_p = convertRawGyro(gz_pinky);

      sample_id++;

      uint8_t buf[153];
      uint8_t *p = buf;

      #define PACK_FLOAT(x)  do { memcpy(p, &(x), 4); p += 4; } while (0) // function to perform mem copy for each variable
      #define PACK_BOOL(b)   do { uint8_t v = (b ? 1 : 0); *p++ = v; } while (0) // for boolean only sigle byte

      memcpy(p, &sample_id, 4); p += 4;

      // base IMU
      PACK_FLOAT(ax_base);
      PACK_FLOAT(ay_base);
      PACK_FLOAT(az_base);
      PACK_FLOAT(gx_base);
      PACK_FLOAT(gy_base);
      PACK_FLOAT(gz_base);

      // thumb
      PACK_FLOAT(ax_tb);
      PACK_FLOAT(ay_tb);
      PACK_FLOAT(az_tb);
      PACK_FLOAT(gx_tb);
      PACK_FLOAT(gy_tb);
      PACK_FLOAT(gz_tb);
      PACK_BOOL(f_thumb);

      // index
      PACK_FLOAT(ax_id);
      PACK_FLOAT(ay_id);
      PACK_FLOAT(az_id);
      PACK_FLOAT(gx_id);
      PACK_FLOAT(gy_id);
      PACK_FLOAT(gz_id);
      PACK_BOOL(f_index);

      // middle
      PACK_FLOAT(ax_m);
      PACK_FLOAT(ay_m);
      PACK_FLOAT(az_m);
      PACK_FLOAT(gx_m);
      PACK_FLOAT(gy_m);
      PACK_FLOAT(gz_m);
      PACK_BOOL(f_middle);

      // ring
      PACK_FLOAT(ax_r);
      PACK_FLOAT(ay_r);
      PACK_FLOAT(az_r);
      PACK_FLOAT(gx_r);
      PACK_FLOAT(gy_r);
      PACK_FLOAT(gz_r);
      PACK_BOOL(f_ring);

      // pinky
      PACK_FLOAT(ax_p);
      PACK_FLOAT(ay_p);
      PACK_FLOAT(az_p);
      PACK_FLOAT(gx_p);
      PACK_FLOAT(gy_p);
      PACK_FLOAT(gz_p);
      PACK_BOOL(f_pinky);

      GIK_tx_Char.writeValue(buf, sizeof(buf));  // send out the packet

      unsigned long elapsed = micros() - startTime; //dynamic delay
      long delayNeeded = 10000 - elapsed;  // set the period here in microseconds
      // Serial.print("Elapsed: ");
      // Serial.print(elapsed);
      // Serial.print("ms, Delay: ");
      // Serial.print(delayNeeded);
      // Serial.println("ms");
      if (delayNeeded > 1000) {
        delay(delayNeeded / 1000);
      }
    }

    Serial.println("Receiver disconnected");
  }
}

float convertRawGyro(int gRaw) {
  // since we are using 250 degrees/seconds range
  // -250 maps to a raw value of -32768
  // +250 maps to a raw value of 32767

  float g = (gRaw * 250.0) / 32768.0;

  return g;
}

float convertRawAccel(int aRaw) {
  // since we are using 250 degrees/seconds range
  // -250 maps to a raw value of -32768
  // +250 maps to a raw value of 32767

  float a = (aRaw * (4/32768.0));

  return a;
}