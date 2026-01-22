#define FSR1_PIN A0
#define FSR2_PIN A1
#define FSR3_PIN A2
#define FSR4_PIN A3
#define FSR5_PIN A6
#define THRESHOLD 600

void setup() {
  Serial.begin(9600);
}

void loop() {
  int FSR1 = analogRead(FSR1_PIN) > THRESHOLD ? 1 : 0;
  int FSR2 = analogRead(FSR2_PIN) > THRESHOLD ? 1 : 0;
  int FSR3 = analogRead(FSR3_PIN) > THRESHOLD ? 1 : 0;
  int FSR4 = analogRead(FSR4_PIN) > THRESHOLD ? 1 : 0;
  int FSR5 = analogRead(FSR5_PIN) > THRESHOLD ? 1 : 0;
  
  Serial.print(FSR1);
  Serial.print(",");
  Serial.print(FSR2);
  Serial.print(",");
  Serial.print(FSR3);
  Serial.print(",");
  Serial.print(FSR4);
  Serial.print(",");
  Serial.println(FSR5);
  delay(100);
}
