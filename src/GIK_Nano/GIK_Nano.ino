//-------------------------------------------------------------------------------------------------------------------
// This code is to make the Arduino Nano BLE Sense Rev 2 on GIK Left Hand to collect the data from all the sensors
// The sensor values are sent to the receiver via Bluetooth service with the MTU size of 153 bytes
//--------------------------------------------------------------------------------------------------------------------

#define LEFT_HAND  // Define this as left hand 
//#define RIGHT_HAND  // Define this as right hand
#include "GIK_Hand_Config.h"  // Include the hand configuration file
#include <ArduinoBLE.h>
#include "Arduino_BMI270_BMM150.h"

// For Finger IMU libraries
#include <SPI.h>
#include <Adafruit_BMP280.h>
#include <BMI160Gen.h>

BLEService GIK_Service(ServiceID);  // service ID
BLECharacteristic GIK_tx_Char(CharID,BLERead | BLENotify,153); // Characteristic ID with notification and MTU of 153 BYTES
Adafruit_BMP280 BMP280;


const char* Name = Hand_Name;

const int CS_THUMB = 10; // Chip Select pin for talking to thumb IMU
const int CS_INDEX = 9; // Chip Select  pin for talking to index IMU
const int CS_MIDDLE = 4; // Chip Select  pin for talking to middle IMU
const int CS_RING = 5; // Chip Select  pin for talking to ring IMU
const int CS_PINKY = 6; // Chip Select  pin for talking to pinky IMU

// Definition of variables for left hand

float ax_base = 0, ay_base = 0, az_base = 0; // accelerometer xyz from left base 
float gx_base = 0, gy_base = 0, gz_base = 0; // gyro xyz from left base


float ax_tb, ay_tb, az_tb, gx_tb, gy_tb, gz_tb;
int ax_thumb, ay_thumb, az_thumb; // accelerometer xyz from left thumb 
int gx_thumb, gy_thumb, gz_thumb; // gyro xyz from left thumb
bool f_thumb = 0; // force sensor boolean for left thumb


float ax_id, ay_id, az_id, gx_id, gy_id, gz_id;
int ax_index = 0, ay_index = 0, az_index = 0; // accelerometer xyz from left index
int gx_index = 0, gy_index = 0, gz_index = 0; // gyro xyz from left index
bool f_index = 0; // force sensor boolean for left index

float ax_middle = 0, ay_middle = 0, az_middle = 0; // accelerometer xyz from left midlle 
float gx_middle = 0, gy_middle = 0, gz_middle = 0; // gyro xyz from left midlle
bool f_middle = 0; // force sensor boolean for left middle

float ax_ring = 0, ay_ring = 0, az_ring = 0; // accelerometer xyz from left ring 
float gx_ring = 0, gy_ring = 0, gz_ring = 0;  // gyro xyz from left ring
bool f_ring = 0; // force sensor boolean for left ring

float ax_pinky = 0, ay_pinky = 0, az_pinky = 0; // accelerometer xyz from left pinky
float gx_pinky = 0, gy_pinky = 0, gz_pinky = 0; // gyro xyz from left pinky
bool f_pinky = 0; // force sensor boolean for left pinky

// Packet layout (little-endian):
// uint32  sample_id
// float   ax_base, ay_base, az_base, gx_base, gy_base, gz_base
// for each finger: thumb, index, middle, ring, pinky
//   float ax, ay, az, gx, gy, gz
//   uint8 f which should add up to 153 bytes of MTU


uint32_t sample_id = 0; // to label the packets

void setup() {
  Serial.begin(9600);

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
  BMI160.begin(CS_THUMB);
  BMI160.setGyroRange(250); // supported values: 125, 250, 500, 1000, 2000 (degrees/second)
  BMI160.setAccelerometerRange(4); // supported values: 2, 4, 8, 16 (G)

  BLE.setLocalName(Name);
  BLE.setAdvertisedService(GIK_Service);

  GIK_Service.addCharacteristic(GIK_tx_Char);
  BLE.addService(GIK_Service);

  BLE.advertise();
  Serial.println("Advertising, waiting for receiver to activate...");

  // set the on board RGB as output (high turn off the RGB, low turn them on)
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);

  pinMode(CS_THUMB, OUTPUT); digitalWrite(CS_THUMB, HIGH);
  pinMode(CS_INDEX, OUTPUT); digitalWrite(CS_INDEX, HIGH);
  pinMode(CS_MIDDLE, OUTPUT); digitalWrite(CS_MIDDLE, HIGH);
  pinMode(CS_RING, OUTPUT); digitalWrite(CS_RING, HIGH);
  pinMode(CS_PINKY, OUTPUT); digitalWrite(CS_PINKY, HIGH);
}

void loop() {

  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  digitalWrite(LEDR, LOW);
  BLEDevice central = BLE.central();   // Wait here until receiver.py connects

  if (central) {
    Serial.print("Connected to Receiver: ");
    Serial.println(central.address());
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDB, HIGH);
    digitalWrite(LEDG, LOW);

    // Reset sample counter on each new connection
    sample_id = 0;

    // Only acquire IMU data and send while connected
    while (central.connected()) {
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(ax_base, ay_base, az_base);
      }

      if (IMU.gyroscopeAvailable()) {
        IMU.readGyroscope(gx_base, gy_base, gz_base);
      }
      // SPI read from thumb IMU

      digitalWrite(CS_THUMB, LOW);
      delay(1);
      BMI160.readMotionSensor(ax_thumb, ay_thumb, az_thumb, gx_thumb, gy_thumb, gz_thumb); //IMU sensor readings from the thumb IMU
      digitalWrite(CS_THUMB, HIGH);
      delay(1);
      digitalWrite(CS_INDEX, LOW);
      delay(1);
      BMI160.readMotionSensor(ax_index, ay_index, az_index, gx_index, gy_index, gz_index); //IMU sensor readings from the thumb IMU
      digitalWrite(CS_INDEX, HIGH);


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
      PACK_FLOAT(ax_middle);
      PACK_FLOAT(ay_middle);
      PACK_FLOAT(az_middle);
      PACK_FLOAT(gx_middle);
      PACK_FLOAT(gy_middle);
      PACK_FLOAT(gz_middle);
      PACK_BOOL(f_middle);

      // ring
      PACK_FLOAT(ax_ring);
      PACK_FLOAT(ay_ring);
      PACK_FLOAT(az_ring);
      PACK_FLOAT(gx_ring);
      PACK_FLOAT(gy_ring);
      PACK_FLOAT(gz_ring);
      PACK_BOOL(f_ring);

      // pinky
      PACK_FLOAT(ax_pinky);
      PACK_FLOAT(ay_pinky);
      PACK_FLOAT(az_pinky);
      PACK_FLOAT(gx_pinky);
      PACK_FLOAT(gy_pinky);
      PACK_FLOAT(gz_pinky);
      PACK_BOOL(f_pinky);


      GIK_tx_Char.writeValue(buf, sizeof(buf));  // send out the packet

      // debug
      Serial.print("id=");
      Serial.print(sample_id);
      Serial.print(" acc_base=");
      Serial.print(ax_tb); Serial.print(",");
      Serial.print(ay_tb); Serial.print(",");
      Serial.print(az_tb);
      Serial.print(" gyro_base=");
      Serial.print(gx_tb); Serial.print(",");
      Serial.print(gy_tb); Serial.print(",");
      Serial.print("id=");
      Serial.print(sample_id);
      Serial.print(" acc_base=");
      Serial.print(ax_id); Serial.print(",");
      Serial.print(ay_id); Serial.print(",");
      Serial.print(az_id);
      Serial.print(" gyro_base=");
      Serial.print(gx_id); Serial.print(",");
      Serial.print(gy_id); Serial.print(",");
      Serial.println(gz_id);

      delay(10);  // ~100 Hz
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