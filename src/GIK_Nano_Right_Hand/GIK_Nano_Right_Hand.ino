//-------------------------------------------------------------------------------------------------------------------
// This code is to make the Arduino Nano BLE Sense Rev 2 on GIK right Hand to collect the data from all the sensors
// The sensor values are sent to the receiver via Bluetooth service with the MTU size of 153 bytes
//--------------------------------------------------------------------------------------------------------------------

#include <ArduinoBLE.h>
#include "Arduino_BMI270_BMM150.h"

BLEService GIK_Service("1236");  // service ID
BLECharacteristic GIK_tx_Char("1237",BLERead | BLENotify,153); // Characteristic ID with notification and MTU of 153 BYTES

const char* NameR = "GIK_Nano_R";

// Definition of variables for right hand

float ax_r_base, ay_r_base, az_r_base; // accelerometer xyz from right base 
float gx_r_base, gy_r_base, gz_r_base; // gyro xyz from right base

float ax_r_thumb = 0, ay_r_thumb = 0, az_r_thumb = 0; // accelerometer xyz from right thumb 
float gx_r_thumb = 0, gy_r_thumb = 0, gz_r_thumb = 0; // gyro xyz from right thumb
bool f_r_thumb = 0; // force sensor boolean for right thumb

float ax_r_index = 0, ay_r_index = 0, az_r_index = 0; // accelerometer xyz from right index
float gx_r_index = 0, gy_r_index = 0, gz_r_index = 0; // gyro xyz from right index
bool f_r_index = 0; // force sensor boolean for right index

float ax_r_middle = 0, ay_r_middle = 0, az_r_middle = 0; // accelerometer xyz from right midlle 
float gx_r_middle = 0, gy_r_middle = 0, gz_r_middle = 0; // gyro xyz from right midlle
bool f_r_middle = 0; // force sensor boolean for right middle

float ax_r_ring = 0, ay_r_ring = 0, az_r_ring = 0; // accelerometer xyz from right ring 
float gx_r_ring = 0, gy_r_ring = 0, gz_r_ring = 0;  // gyro xyz from right ring
bool f_r_ring = 0; // force sensor boolean for right ring

float ax_r_pinky = 0, ay_r_pinky = 0, az_r_pinky = 0; // accelerometer xyz from right pinky
float gx_r_pinky = 0, gy_r_pinky = 0, gz_r_pinky = 0; // gyro xyz from right pinky
bool f_r_pinky = 0; // force sensor boolean for right pinky

// Packet layout (little-endian):
// uint32  sample_id
// float   ax_r_base, ay_r_base, az_r_base, gx_r_base, gy_r_base, gz_r_base
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

  BLE.setLocalName(NameR);
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
      if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        IMU.readAcceleration(ax_r_base, ay_r_base, az_r_base);
        IMU.readGyroscope(gx_r_base, gy_r_base, gz_r_base);

        sample_id++;

        uint8_t buf[153];
        uint8_t *p = buf;

        #define PACK_FLOAT(x)  do { memcpy(p, &(x), 4); p += 4; } while (0) // function to perform mem copy for each variable
        #define PACK_BOOL(b)   do { uint8_t v = (b ? 1 : 0); *p++ = v; } while (0) // for boolean only sigle byte

        memcpy(p, &sample_id, 4); p += 4;

        // base IMU
        PACK_FLOAT(ax_r_base);
        PACK_FLOAT(ay_r_base);
        PACK_FLOAT(az_r_base);
        PACK_FLOAT(gx_r_base);
        PACK_FLOAT(gy_r_base);
        PACK_FLOAT(gz_r_base);

        // thumb
        PACK_FLOAT(ax_r_thumb);
        PACK_FLOAT(ay_r_thumb);
        PACK_FLOAT(az_r_thumb);
        PACK_FLOAT(gx_r_thumb);
        PACK_FLOAT(gy_r_thumb);
        PACK_FLOAT(gz_r_thumb);
        PACK_BOOL(f_r_thumb);

        // index
        PACK_FLOAT(ax_r_index);
        PACK_FLOAT(ay_r_index);
        PACK_FLOAT(az_r_index);
        PACK_FLOAT(gx_r_index);
        PACK_FLOAT(gy_r_index);
        PACK_FLOAT(gz_r_index);
        PACK_BOOL(f_r_index);

        // middle
        PACK_FLOAT(ax_r_middle);
        PACK_FLOAT(ay_r_middle);
        PACK_FLOAT(az_r_middle);
        PACK_FLOAT(gx_r_middle);
        PACK_FLOAT(gy_r_middle);
        PACK_FLOAT(gz_r_middle);
        PACK_BOOL(f_r_middle);

        // ring
        PACK_FLOAT(ax_r_ring);
        PACK_FLOAT(ay_r_ring);
        PACK_FLOAT(az_r_ring);
        PACK_FLOAT(gx_r_ring);
        PACK_FLOAT(gy_r_ring);
        PACK_FLOAT(gz_r_ring);
        PACK_BOOL(f_r_ring);

        // pinky
        PACK_FLOAT(ax_r_pinky);
        PACK_FLOAT(ay_r_pinky);
        PACK_FLOAT(az_r_pinky);
        PACK_FLOAT(gx_r_pinky);
        PACK_FLOAT(gy_r_pinky);
        PACK_FLOAT(gz_r_pinky);
        PACK_BOOL(f_r_pinky);


        GIK_tx_Char.writeValue(buf, sizeof(buf));  // send out the packet

        // // debug
        // Serial.print("id=");
        // Serial.print(sample_id);
        // Serial.print(" acc_r_base=");
        // Serial.print(ax_r_base); Serial.print(",");
        // Serial.print(ay_r_base); Serial.print(",");
        // Serial.print(az_r_base);
        // Serial.print(" gyro_r_base=");
        // Serial.print(gx_r_base); Serial.print(",");
        // Serial.print(gy_r_base); Serial.print(",");
        // Serial.println(gz_r_base);
      }

      delay(10);  // ~100 Hz
    }

    Serial.println("Receiver disconnected");
  }
}
