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
#define THRESHOLD 5
#define THRESHOLDLOW 3

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

// Base IMU biases
float ax_base_bias = 0, ay_base_bias = 0, az_base_bias = 0;
float gx_base_bias = 0, gy_base_bias = 0, gz_base_bias = 0;

// Finger IMU biases (float, even though raw is int)
float ax_tb_bias = 0, ay_tb_bias = 0, az_tb_bias = 0;
float gx_tb_bias = 0, gy_tb_bias = 0, gz_tb_bias = 0;

float ax_id_bias = 0, ay_id_bias = 0, az_id_bias = 0;
float gx_id_bias = 0, gy_id_bias = 0, gz_id_bias = 0;

float ax_m_bias  = 0, ay_m_bias  = 0, az_m_bias  = 0;
float gx_m_bias  = 0, gy_m_bias  = 0, gz_m_bias  = 0;

float ax_r_bias  = 0, ay_r_bias  = 0, az_r_bias  = 0;
float gx_r_bias  = 0, gy_r_bias  = 0, gz_r_bias  = 0;

float ax_p_bias  = 0, ay_p_bias  = 0, az_p_bias  = 0;
float gx_p_bias  = 0, gy_p_bias  = 0, gz_p_bias  = 0;


void calibrateIMUs(unsigned long calib_ms = 3000) {
  Serial.println("Starting IMU calibration... keep hand still.");

  unsigned long start = millis();
  unsigned long count = 0;

  // Sums for base IMU
  double axb_sum = 0, ayb_sum = 0, azb_sum = 0;
  double gxb_sum = 0, gyb_sum = 0, gzb_sum = 0;

  // Sums for fingers (use float outputs from convertRaw*)
  double axtb_sum = 0, aytb_sum = 0, aztb_sum = 0;
  double gxtb_sum = 0, gytb_sum = 0, gztb_sum = 0;

  double axid_sum = 0, ayid_sum = 0, azid_sum = 0;
  double gxid_sum = 0, gyid_sum = 0, gzid_sum = 0;

  double axm_sum  = 0, aym_sum  = 0, azm_sum  = 0;
  double gxm_sum  = 0, gym_sum  = 0, gzm_sum  = 0;

  double axr_sum  = 0, ayr_sum  = 0, azr_sum  = 0;
  double gxr_sum  = 0, gyr_sum  = 0, gzr_sum  = 0;

  double axp_sum  = 0, ayp_sum  = 0, azp_sum  = 0;
  double gxp_sum  = 0, gyp_sum  = 0, gzp_sum  = 0;

  int ax_ti, ay_ti, az_ti, gx_ti, gy_ti, gz_ti; // temp raw ints

  while (millis() - start < calib_ms) {
    // base IMU
    if (IMU.accelerationAvailable()) {
      float axb, ayb, azb;
      IMU.readAcceleration(axb, ayb, azb);
      axb_sum += axb;
      ayb_sum += ayb;
      azb_sum += azb;
    }
    if (IMU.gyroscopeAvailable()) {
      float gxb, gyb, gzb;
      IMU.readGyroscope(gxb, gyb, gzb);
      gxb_sum += gxb;
      gyb_sum += gyb;
      gzb_sum += gzb;
    }

    // thumb
    digitalWrite(CS_THUMB, LOW);
    BMI160.readMotionSensor(ax_ti, ay_ti, az_ti, gx_ti, gy_ti, gz_ti);
    digitalWrite(CS_THUMB, HIGH);
    axtb_sum += convertRawAccel(ax_ti);
    aytb_sum += convertRawAccel(ay_ti);
    aztb_sum += convertRawAccel(az_ti);
    gxtb_sum += convertRawGyro(gx_ti);
    gytb_sum += convertRawGyro(gy_ti);
    gztb_sum += convertRawGyro(gz_ti);

    // index
    digitalWrite(CS_INDEX, LOW);
    BMI160.readMotionSensor(ax_ti, ay_ti, az_ti, gx_ti, gy_ti, gz_ti);
    digitalWrite(CS_INDEX, HIGH);
    axid_sum += convertRawAccel(ax_ti);
    ayid_sum += convertRawAccel(ay_ti);
    azid_sum += convertRawAccel(az_ti);
    gxid_sum += convertRawGyro(gx_ti);
    gyid_sum += convertRawGyro(gy_ti);
    gzid_sum += convertRawGyro(gz_ti);

    // middle
    digitalWrite(CS_MIDDLE, LOW);
    BMI160.readMotionSensor(ax_ti, ay_ti, az_ti, gx_ti, gy_ti, gz_ti);
    digitalWrite(CS_MIDDLE, HIGH);
    axm_sum += convertRawAccel(ax_ti);
    aym_sum += convertRawAccel(ay_ti);
    azm_sum += convertRawAccel(az_ti);
    gxm_sum += convertRawGyro(gx_ti);
    gym_sum += convertRawGyro(gy_ti);
    gzm_sum += convertRawGyro(gz_ti);

    // ring
    digitalWrite(CS_RING, LOW);
    BMI160.readMotionSensor(ax_ti, ay_ti, az_ti, gx_ti, gy_ti, gz_ti);
    digitalWrite(CS_RING, HIGH);
    axr_sum += convertRawAccel(ax_ti);
    ayr_sum += convertRawAccel(ay_ti);
    azr_sum += convertRawAccel(az_ti);
    gxr_sum += convertRawGyro(gx_ti);
    gyr_sum += convertRawGyro(gy_ti);
    gzr_sum += convertRawGyro(gz_ti);

    // pinky
    digitalWrite(CS_PINKY, LOW);
    BMI160.readMotionSensor(ax_ti, ay_ti, az_ti, gx_ti, gy_ti, gz_ti);
    digitalWrite(CS_PINKY, HIGH);
    axp_sum += convertRawAccel(ax_ti);
    ayp_sum += convertRawAccel(ay_ti);
    azp_sum += convertRawAccel(az_ti);
    gxp_sum += convertRawGyro(gx_ti);
    gyp_sum += convertRawGyro(gy_ti);
    gzp_sum += convertRawGyro(gz_ti);

    count++;
    delay(5); // small delay to avoid spamming
  }

  if (count == 0) return;

  // Base IMU biases
  ax_base_bias = axb_sum / count;
  ay_base_bias = ayb_sum / count;
  az_base_bias = azb_sum / count;  // includes gravity if z-axis aligned
  gx_base_bias = gxb_sum / count;
  gy_base_bias = gyb_sum / count;
  gz_base_bias = gzb_sum / count;

  // Thumb
  ax_tb_bias = axtb_sum / count;
  ay_tb_bias = aytb_sum / count;
  az_tb_bias = aztb_sum / count;
  gx_tb_bias = gxtb_sum / count;
  gy_tb_bias = gytb_sum / count;
  gz_tb_bias = gztb_sum / count;

  // Index
  ax_id_bias = axid_sum / count;
  ay_id_bias = ayid_sum / count;
  az_id_bias = azid_sum / count;
  gx_id_bias = gxid_sum / count;
  gy_id_bias = gyid_sum / count;
  gz_id_bias = gzid_sum / count;

  // Middle
  ax_m_bias = axm_sum / count;
  ay_m_bias = aym_sum / count;
  az_m_bias = azm_sum / count;
  gx_m_bias = gxm_sum / count;
  gy_m_bias = gym_sum / count;
  gz_m_bias = gzm_sum / count;

  // Ring
  ax_r_bias = axr_sum / count;
  ay_r_bias = ayr_sum / count;
  az_r_bias = azr_sum / count;
  gx_r_bias = gxr_sum / count;
  gy_r_bias = gyr_sum / count;
  gz_r_bias = gzr_sum / count;

  // Pinky
  ax_p_bias = axp_sum / count;
  ay_p_bias = ayp_sum / count;
  az_p_bias = azp_sum / count;
  gx_p_bias = gxp_sum / count;
  gy_p_bias = gyp_sum / count;
  gz_p_bias = gzp_sum / count;

}



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

    // Reset sample counter on each new connection
    sample_id = 0;

    while (!GIK_tx_Char.subscribed() && central.connected()) {
      delay(100); // Wait until receiver.py subscribes then only move to data sending to prevent sample index loss
    }

    digitalWrite(LEDR, HIGH);  // blue light
    digitalWrite(LEDB, LOW);
    digitalWrite(LEDG, HIGH);

    calibrateIMUs(3000);


    for (int i = 0; i < 10; ++i) {
      digitalWrite(LEDR, HIGH);  // alert the user to ready
      digitalWrite(LEDB, HIGH);
      digitalWrite(LEDG, HIGH);
      delay(200); 
      digitalWrite(LEDR, HIGH); 
      digitalWrite(LEDB, LOW);
      digitalWrite(LEDG, HIGH);
      delay(200);
    }

    delay(500);

    int last_thumb = analogRead(FSR1_PIN);
    int last_index = analogRead(FSR2_PIN);
    int last_middle = analogRead(FSR3_PIN);
    int last_ring = analogRead(FSR4_PIN);
    int last_pinky = analogRead(FSR5_PIN);
    while (central.connected()) {

      digitalWrite(LEDR, HIGH);
      digitalWrite(LEDB, HIGH);
      digitalWrite(LEDG, LOW);
  

      unsigned long startTime = micros();


      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(ax_base, ay_base, az_base);
      }

      if (IMU.gyroscopeAvailable()) {
        IMU.readGyroscope(gx_base, gy_base, gz_base);
      }

      int current_thumb = analogRead(FSR1_PIN);
      bool f_thumb = current_thumb - last_thumb > THRESHOLDLOW ? 1 : 0;
      int current_index = analogRead(FSR2_PIN);
      bool f_index = current_index - last_index > THRESHOLDLOW ? 1 : 0;
      int current_middle = analogRead(FSR3_PIN);
      bool f_middle = current_middle - last_middle > THRESHOLDLOW ? 1 : 0;
      int current_ring = analogRead(FSR4_PIN);
      bool f_ring = current_ring - last_ring > THRESHOLDLOW ? 1 : 0;
      int current_pinky = analogRead(FSR5_PIN);
      bool f_pinky = current_pinky - last_pinky > THRESHOLDLOW ? 1 : 0;

      last_thumb = current_thumb;
      last_index = current_index;
      last_middle = current_middle;
      last_ring = current_ring;
      last_pinky = current_pinky;

      // Serial.print("thumb=");  Serial.print(analogRead(FSR1_PIN));
      // Serial.print(" index="); Serial.print(analogRead(FSR2_PIN));
      // Serial.print(" middle=");Serial.print(analogRead(FSR3_PIN));
      // Serial.print(" ring=");  Serial.print(analogRead(FSR4_PIN));
      // Serial.print(" pinky="); Serial.println(analogRead(FSR5_PIN));
      // Serial.print("thumb=");  Serial.print(f_thumb);
      // Serial.print(" index="); Serial.print(f_index);
      // Serial.print(" middle=");Serial.print(f_middle);
      // Serial.print(" ring=");  Serial.print(f_ring);
      // Serial.print(" pinky="); Serial.println(f_pinky);

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


      ax_base = ax_base - ax_base_bias;
      ay_base = ay_base - ay_base_bias;
      az_base = az_base - az_base_bias;
      gx_base = gx_base - gx_base_bias;
      gy_base = gy_base - gy_base_bias;
      gz_base = gz_base - gz_base_bias;


      ax_tb = convertRawAccel(ax_thumb) - ax_tb_bias;
      ay_tb = convertRawAccel(ay_thumb) - ay_tb_bias;
      az_tb = convertRawAccel(az_thumb) - az_tb_bias;
      gx_tb = convertRawGyro(gx_thumb) - gx_tb_bias;
      gy_tb = convertRawGyro(gy_thumb) - gy_tb_bias;
      gz_tb = convertRawGyro(gz_thumb) - gz_tb_bias;

      ax_id = convertRawAccel(ax_index) - ax_id_bias;
      ay_id = convertRawAccel(ay_index) - ay_id_bias;
      az_id = convertRawAccel(az_index) - az_id_bias;
      gx_id = convertRawGyro(gx_index) - gx_id_bias;
      gy_id = convertRawGyro(gy_index) - gy_id_bias;
      gz_id = convertRawGyro(gz_index) - gz_id_bias;

      ax_m = convertRawAccel(ax_middle) - ax_m_bias;
      ay_m = convertRawAccel(ay_middle) - ay_m_bias;
      az_m = convertRawAccel(az_middle) - az_m_bias;
      gx_m = convertRawGyro(gx_middle) - gx_m_bias;
      gy_m = convertRawGyro(gy_middle) - gy_m_bias;
      gz_m = convertRawGyro(gz_middle) - gz_m_bias;

      ax_r = convertRawAccel(ax_ring) - ax_r_bias;
      ay_r = convertRawAccel(ay_ring) - ay_r_bias;
      az_r = convertRawAccel(az_ring) - az_r_bias;
      gx_r = convertRawGyro(gx_ring) - gx_r_bias;
      gy_r = convertRawGyro(gy_ring) - gy_r_bias;
      gz_r = convertRawGyro(gz_ring) - gz_r_bias;

      ax_p = convertRawAccel(ax_pinky) - ax_p_bias;
      ay_p = convertRawAccel(ay_pinky) - ay_p_bias;
      az_p = convertRawAccel(az_pinky) - az_p_bias;
      gx_p = convertRawGyro(gx_pinky) - gx_p_bias;
      gy_p = convertRawGyro(gy_pinky) - gy_p_bias;
      gz_p = convertRawGyro(gz_pinky) - gz_p_bias;

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
      long delayNeeded = 35000 - elapsed;  // set the period here in microseconds
      
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
