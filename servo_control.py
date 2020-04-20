import RPi.GPIO as GPIO
import time

"""
Class written for simple servo control with RaspPi. Written by
Ethan Lyon for our smart_lock project in ELEC 574.

Initialize a servo object with desired parameters,
then run Servo.setup() to start pwm.
Change angles with Servo.rotate and lock angles with
Servo.lock_angle().

It is recommended to add a wait time between rotations
in order to allow the Servo time to reach the target angle.
Locking the angle between rotations is highly recommended to
avoid jitter and increse Servo stability.

Run Servo.stop() at the end of the program to clean up.
"""

class Servo:
    def __init__(self, pin, freq = 50, angle = 90, pulse_start = 0.5 , pulse_end = 2.4):
        self.freq = freq
        self.angle = angle
        self.pin = pin
        self.pulse_start = pulse_start
        self.pulse_end = pulse_end
        self.pwm = None
        
    def calculate_duty_percent(self, ang):
        #Calculate the PWM duty cycle percent corresponding to the desired angle
        print(ang)
        pdel = self.pulse_end - self.pulse_start
        assert (pdel > 0), "Servo pulse width must be positive."
        duty_cycle = 1000/self.freq
        assert ((ang >= 0) and (ang <= 180)), "Invalid Angle: Servo angles are between 0 and 180 degrees"
        pw = ang*pdel/180 + self.pulse_start
        duty_percent = 100*pw/duty_cycle
        return duty_percent
    
    def setup(self):
        #Set up GPIO and start PWM
        GPIO.cleanup()
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, self.freq)
        self.pwm.start(self.calculate_duty_percent(self.angle))
        
    def rotate(self, ang):
        #Rotate the servo to the desired angle
        self.angle = ang;
        assert self.pwm is not None, "PWM is not initialized correctly. Run Servo.setup() to continue."
        duty_percent = self.calculate_duty_percent(ang)
        print(duty_percent)
        self.pwm.ChangeDutyCycle(duty_percent)
        time.sleep(0.5)
        
    def lock_angle(self, wait_time = 0):
        #Set the pwm duty cycle to 0 in order to prevent jitter and increase stability
        if(wait_time == 0):
            self.pwm.ChangeDutyCycle(0)
        else:
            time.sleep(wait_time) #Give the servo time to reach the desired angle before locking it
            self.pwm.ChangeDutyCycle(0)
            
    def rotate_and_lock(self, ang, wait_time = 0):
        #Rotate Servo to angle and lock it
        if(wait_time == 0):
            #Not having a wait time is not recommended
            self.rotate(ang)
            self.lock_angle()
        else:
            self.rotate(ang)
            time.sleep(wait_time)
            self.lock_angle()
    
    def stop(self):
        #Stop servo and end PWM and cleanup GPIO
        self.pwm.stop()
        del self.pwm
        GPIO.cleanup()
   
   
#Uncomment to test the servo   
"""
pulse_start_width = 0.5
pulse_end_width = 2.4
SERVO_PIN = 17

s = Servo(SERVO_PIN, angle = 0, pulse_start = pulse_start_width, pulse_end = pulse_end_width)
s.setup()

s.rotate(0); s.lock_angle(wait_time = 0.5)
s.rotate(45); s.lock_angle(wait_time = 0.5)
s.rotate(90); s.lock_angle(wait_time = 0.5)
s.rotate(135); s.lock_angle(wait_time = 0.5)
s.rotate(180); s.lock_angle(wait_time = 0.5)
s.rotate(90); s.lock_angle(wait_time = 0.5)


s.rotate_and_lock(90, wait_time = 0.5)
s.rotate_and_lock(45, wait_time = 0.5)
s.rotate_and_lock(135, wait_time = 0.5)
s.rotate_and_lock(180, wait_time = 0.5)
s.stop()
"""
        