#define vibPin A0
#define pirPin 7
unsigned long previousMillis = 0; 
const unsigned long interval = 400;

void setup(){
  Serial.begin(9600);
  pinMode(vibPin,INPUT);
  pinMode(pirPin,INPUT);
}
void loop(){
  int vibData = (analogRead(vibPin)>500) ? 1:0;  
  int pirData = digitalRead(pirPin);
unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {  
    previousMillis = currentMillis;
  Serial.print("Data,");
  Serial.print(vibData);
  Serial.print(", ");
  Serial.println(pirData);
  }

  
}
