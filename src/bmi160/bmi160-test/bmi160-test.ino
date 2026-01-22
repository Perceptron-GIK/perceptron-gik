#include <SPI.h>
#include <Adafruit_BMP280.h>
#include <BMI160Gen.h>

const int select_pin = 10;
Adafruit_BMP280 BMP280;

void setup() {
  Serial.begin(9600);
  Serial.println("Initialising BMI160...");
  while (!Serial); 

  SPI.begin();
  BMI160.begin(select_pin);
  uint8_t dev_id = BMI160.getDeviceID();
  Serial.print("DEVICE ID: ");
  Serial.println(dev_id, HEX);

  BMI160.setGyroRange(250);
  BMI160.setAccelerometerRange(4);
}

void loop() {
  int axRaw, ayRaw, azRaw, gxRaw, gyRaw, gzRaw, mxRaw, myRaw, mzRaw, tRaw;
  float ax, ay, az, gx, gy, gz;

  BMI160.readMotionSensor9(axRaw, ayRaw, azRaw, gxRaw, gyRaw, gzRaw, mxRaw, myRaw, mzRaw, tRaw);
  
  gx = convertRawGyro(gxRaw);
  gy = convertRawGyro(gyRaw);
  gz = convertRawGyro(gzRaw);

  ax = convertRawAccel(axRaw);
  ay = convertRawAccel(ayRaw);
  az = convertRawAccel(azRaw);

  Serial.print("a:\t");
  Serial.print(ax);
  Serial.print("\t");
  Serial.print(ay);
  Serial.print("\t");
  Serial.print(az);
  Serial.println();
  Serial.print("g:\t");
  Serial.print(gx);
  Serial.print("\t");
  Serial.print(gy);
  Serial.print("\t");
  Serial.print(gz);
  Serial.println();
  Serial.print("m:\t");
  Serial.print(mxRaw);
  Serial.print("\t");
  Serial.print(myRaw);
  Serial.print("\t");
  Serial.print(mzRaw);
  Serial.println();
  Serial.print(tRaw);
  Serial.println();
  Serial.println();

  delay(500);
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
