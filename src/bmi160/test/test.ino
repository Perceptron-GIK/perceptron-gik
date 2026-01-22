#include <BMI160Gen.h>
#include <Wire.h>

const int i2c_addr = 0x68; 

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Wire.begin();

  if (!BMI160.begin(BMI160GenClass::I2C_MODE, i2c_addr)) {
    Serial.println("BMI160 initialization failed!");
    while (1);
  }

  Serial.println("BMI160 initialized successfully in I2C mode!");
}

void loop() {
  int gx, gy, gz;
  int ax, ay, az;

  BMI160.readGyro(gx, gy, gz);
  BMI160.readAccelerometer(ax, ay, az);

  Serial.print("Gyroscope Data (X, Y, Z): ");
  Serial.print(gx);
  Serial.print(", ");
  Serial.print(gy);
  Serial.print(", ");
  Serial.println(gz);

  Serial.print("Accelerometer Data (X, Y, Z): ");
  Serial.print(ax);
  Serial.print(", ");
  Serial.print(ay);
  Serial.print(", ");
  Serial.println(az);

  delay(100);
}