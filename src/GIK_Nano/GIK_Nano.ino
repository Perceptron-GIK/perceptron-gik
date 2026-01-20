//-------------------------------------------------------------------------------------------------------------------
// This code is to make the Arduino Nano BLE Sense Rev 2 on GIK Left Hand to collect the data from all the sensors
// The sensor values are sent to the receiver via Bluetooth service with the MTU size of 153 bytes
//--------------------------------------------------------------------------------------------------------------------

#define LEFT_HAND  // Define this as left hand 
//#define RIGHT_HAND  // Define this as right hand
#include "GIK_Hand_Config.h"  // Include the hand configuration file
#include <ArduinoBLE.h>
#include "Arduino_BMI270_BMM150.h"

BLEService GIK_Service(ServiceID);  // service ID
BLECharacteristic GIK_tx_Char(CharID,BLERead | BLENotify,165); // Characteristic ID with notification and MTU of 153 BYTES

const char* Name = Hand_Name;

// Definition of variables for left hand

float ax_base = 0, ay_base = 0, az_base = 0; // accelerometer xyz from left base 
float gx_base = 0, gy_base = 0, gz_base = 0; // gyro xyz from left base
float mx_base = 0, my_base = 0, mz_base = 0; // magnatometer xyz from left base

float ax_thumb = 0, ay_thumb = 0, az_thumb = 0; // accelerometer xyz from left thumb 
float gx_thumb = 0, gy_thumb = 0, gz_thumb = 0; // gyro xyz from left thumb
bool f_thumb = 0; // force sensor boolean for left thumb

float ax_index = 0, ay_index = 0, az_index = 0; // accelerometer xyz from left index
float gx_index = 0, gy_index = 0, gz_index = 0; // gyro xyz from left index
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
// float   ax_base, ay_base, az_base, gx_base, gy_base, gz_base, mx_base, my_base, mz_base
// for each finger: thumb, index, middle, ring, pinky
//   float ax, ay, az, gx, gy, gz
//   uint8 f which should add up to 165 bytes of MTU


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
        IMU.readAcceleration(ax_base, ay_base, az_base); // Acceleration in g
      }

      if (IMU.gyroscopeAvailable()) {
        IMU.readGyroscope(gx_base, gy_base, gz_base); // Gyro vaules in degrees/second
      }

      if (IMU.magneticFieldAvailable()) {
        IMU.readMagneticField(mx_base, my_base, mz_base); // Magnetic field in the units of: uT
      }

      sample_id++;

      uint8_t buf[165];
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
      PACK_FLOAT(mx_base);
      PACK_FLOAT(my_base);
      PACK_FLOAT(mz_base);

      // thumb
      PACK_FLOAT(ax_thumb);
      PACK_FLOAT(ay_thumb);
      PACK_FLOAT(az_thumb);
      PACK_FLOAT(gx_thumb);
      PACK_FLOAT(gy_thumb);
      PACK_FLOAT(gz_thumb);
      PACK_BOOL(f_thumb);

      // index
      PACK_FLOAT(ax_index);
      PACK_FLOAT(ay_index);
      PACK_FLOAT(az_index);
      PACK_FLOAT(gx_index);
      PACK_FLOAT(gy_index);
      PACK_FLOAT(gz_index);
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

      // // debug
      // Serial.print("id=");
      // Serial.print(sample_id);
      // Serial.print(" acc_base=");
      // Serial.print(ax_base); Serial.print(",");
      // Serial.print(ay_base); Serial.print(",");
      // Serial.print(az_base);
      // Serial.print(" gyro_base=");
      // Serial.print(gx_base); Serial.print(",");
      // Serial.print(gy_base); Serial.print(",");
      // Serial.println(gz_base);

      delay(10);  // ~100 Hz
    }

    Serial.println("Receiver disconnected");
  }
}
