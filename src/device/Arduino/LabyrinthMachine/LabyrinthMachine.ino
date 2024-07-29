/*************************************************** 
  reads the serially transmitted message, divides the read string into individual components 
  and controls the servos according to the specifications.
  The Adafruit drivers use I2C to communicate (2 Pins: SDA, SCL).

@authors: Sandra Lassahn
@contact: http://www.haw-hamburg.de/marc-hensel
@copyright: 2024
@version: 2024.06.27
@license: CC BY-NC-SA 4.0, see https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
 ****************************************************/
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(); //default address 0x40
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates

String command_message, channel_str, pulse_width_str;
char msg[10];
int pulse_min[] = {420, 350};   //This is the 'minimum' pulse length count (out of 4096)
int pulse_max[] = {2540, 2650}; //This is the 'maximum' pulse length count (out of 4096)
//Depending on the used servo

void setup() {
  Serial.begin(115200);  // Start serial communication with baudrate 115200
  Serial.setTimeout(2);  // Timeout in ms - Waiting time until no more characters are read in

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates

  delay(10);
}

void loop() {
  while (!Serial.available());

  //Message lesen 
  command_message = Serial.readString();
  command_message.toCharArray(msg, command_message.length()+1);   // conversion to charArray is needed for strtok

  channel_str = strtok(msg,";"); //Numbers before semicolons
  int channel = atoi(channel_str.c_str()); // convert char array to int
  pulse_width_str = strtok(NULL, ";"); // take out the next number block after the semicolon
  int pulse_width = atoi(pulse_width_str.c_str());

  // Set PWM signal for the servo control
  if(pulse_width >= pulse_min[channel] and pulse_width <= pulse_max[channel]){
    Serial.print(pulse_width);
    pwm.writeMicroseconds(channel, pulse_width);
    delay(20); //Delay time in ms as security for servos
  } 
}
