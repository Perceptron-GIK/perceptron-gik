//-------------------------------------------------------------------------------------------------------------------
// This code is to make the Arduino Nano BLE Sense Rev 2 on GIK Left Hand to collect the data from all the sensors
// The sensor values are sent to the receiver via Bluetooth service with the MTU size of 153 bytes
//--------------------------------------------------------------------------------------------------------------------

#include <ArduinoBLE.h>
#include "Arduino_BMI270_BMM150.h"

BLEService GIK_Service("1234");  // service ID
BLECharacteristic GIK_tx_Char("1235",BLERead | BLENotify,153); // Characteristic ID with notification and MTU of 153 BYTES

const char* NameL = "GIK_Nano_L";

// Definition of variables for left hand

float ax_l_base, ay_l_base, az_l_base; // accelerometer xyz from left base 
float gx_l_base, gy_l_base, gz_l_base; // gyro xyz from left base

float ax_l_thumb = 0, ay_l_thumb = 0, az_l_thumb = 0; // accelerometer xyz from left thumb 
float gx_l_thumb = 0, gy_l_thumb = 0, gz_l_thumb = 0; // gyro xyz from left thumb
bool f_l_thumb = 0; // force sensor boolean for left thumb

float ax_l_index = 0, ay_l_index = 0, az_l_index = 0; // accelerometer xyz from left index
float gx_l_index = 0, gy_l_index = 0, gz_l_index = 0; // gyro xyz from left index
bool f_l_index = 0; // force sensor boolean for left index

float ax_l_middle = 0, ay_l_middle = 0, az_l_middle = 0; // accelerometer xyz from left midlle 
float gx_l_middle = 0, gy_l_middle = 0, gz_l_middle = 0; // gyro xyz from left midlle
bool f_l_middle = 0; // force sensor boolean for left middle

float ax_l_ring = 0, ay_l_ring = 0, az_l_ring = 0; // accelerometer xyz from left ring 
float gx_l_ring = 0, gy_l_ring = 0, gz_l_ring = 0;  // gyro xyz from left ring
bool f_l_ring = 0; // force sensor boolean for left ring

float ax_l_pinky = 0, ay_l_pinky = 0, az_l_pinky = 0; // accelerometer xyz from left pinky
float gx_l_pinky = 0, gy_l_pinky = 0, gz_l_pinky = 0; // gyro xyz from left pinky
bool f_l_pinky = 0; // force sensor boolean for left pinky

// Packet layout (little-endian):
// uint32  sample_id
// float   ax_l_base, ay_l_base, az_l_base, gx_l_base, gy_l_base, gz_l_base
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

  BLE.setLocalName(NameL);
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
        IMU.readAcceleration(ax_l_base, ay_l_base, az_l_base);
        IMU.readGyroscope(gx_l_base, gy_l_base, gz_l_base);

        sample_id++;

        uint8_t buf[153];
        uint8_t *p = buf;

        #define PACK_FLOAT(x)  do { memcpy(p, &(x), 4); p += 4; } while (0) // function to perform mem copy for each variable
        #define PACK_BOOL(b)   do { uint8_t v = (b ? 1 : 0); *p++ = v; } while (0) // for boolean only sigle byte

        memcpy(p, &sample_id, 4); p += 4;

        // base IMU
        PACK_FLOAT(ax_l_base);
        PACK_FLOAT(ay_l_base);
        PACK_FLOAT(az_l_base);
        PACK_FLOAT(gx_l_base);
        PACK_FLOAT(gy_l_base);
        PACK_FLOAT(gz_l_base);

        // thumb
        PACK_FLOAT(ax_l_thumb);
        PACK_FLOAT(ay_l_thumb);
        PACK_FLOAT(az_l_thumb);
        PACK_FLOAT(gx_l_thumb);
        PACK_FLOAT(gy_l_thumb);
        PACK_FLOAT(gz_l_thumb);
        PACK_BOOL(f_l_thumb);

        // index
        PACK_FLOAT(ax_l_index);
        PACK_FLOAT(ay_l_index);
        PACK_FLOAT(az_l_index);
        PACK_FLOAT(gx_l_index);
        PACK_FLOAT(gy_l_index);
        PACK_FLOAT(gz_l_index);
        PACK_BOOL(f_l_index);

        // middle
        PACK_FLOAT(ax_l_middle);
        PACK_FLOAT(ay_l_middle);
        PACK_FLOAT(az_l_middle);
        PACK_FLOAT(gx_l_middle);
        PACK_FLOAT(gy_l_middle);
        PACK_FLOAT(gz_l_middle);
        PACK_BOOL(f_l_middle);

        // ring
        PACK_FLOAT(ax_l_ring);
        PACK_FLOAT(ay_l_ring);
        PACK_FLOAT(az_l_ring);
        PACK_FLOAT(gx_l_ring);
        PACK_FLOAT(gy_l_ring);
        PACK_FLOAT(gz_l_ring);
        PACK_BOOL(f_l_ring);

        // pinky
        PACK_FLOAT(ax_l_pinky);
        PACK_FLOAT(ay_l_pinky);
        PACK_FLOAT(az_l_pinky);
        PACK_FLOAT(gx_l_pinky);
        PACK_FLOAT(gy_l_pinky);
        PACK_FLOAT(gz_l_pinky);
        PACK_BOOL(f_l_pinky);


        GIK_tx_Char.writeValue(buf, sizeof(buf));  // send out the packet

        // // debug
        // Serial.print("id=");
        // Serial.print(sample_id);
        // Serial.print(" acc_l_base=");
        // Serial.print(ax_l_base); Serial.print(",");
        // Serial.print(ay_l_base); Serial.print(",");
        // Serial.print(az_l_base);
        // Serial.print(" gyro_l_base=");
        // Serial.print(gx_l_base); Serial.print(",");
        // Serial.print(gy_l_base); Serial.print(",");
        // Serial.println(gz_l_base);
      }

      delay(10);  // ~100 Hz
    }

    Serial.println("Receiver disconnected");
  }
}
