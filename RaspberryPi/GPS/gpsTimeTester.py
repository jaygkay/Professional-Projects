''' GPS time check '''

''' Moodules and assigned variables '''
# GPS data
import serial
gps = serial.Serial("/dev/ttyUSB0", baudrate = 9600, timeout=10000)

# Time data
from datetime import datetime
from pytz import timezone
ts = "%Y-%m-%d %H:%M:%S"

# Sensor data
from sense_hat import SenseHat
sense = SenseHat()
sense.clear()

# Button 
from button import *

message = {}
''' Collecting the data '''
while True:
        data_line = gps.readline()
        data = data_line.split(',')
        butoon = Button(25)
	button = Button(25, debounce = 1.0)
	buttonPress = button.is_pressed()
#	message = {}

	# GPS data (location, speed)
	if data[0] == '$GPRMC':
		
                print("****first data****")
#               print("gps time",data[1])
#		print('location:',(data[3], data[5]))
#		print('speed', data[7])
                
		chi_tz = datetime.now(timezone('America/Chicago'))
        	chi_dt = chi_tz.strftime(ts)
		message['first'] = chi_dt
#        	print("1st time",chi_dt)

		# Sensor data
		pitch, roll, yaw = sense.get_orientation().values()
                ax, ay, az = sense.get_accelerometer_raw().values()
                mx, my, mz = sense.get_compass_raw().values()
#               print(pitch, roll, yaw, ax, ay, az,mx, my, mz)

                chi_tz = datetime.now(timezone('America/Chicago'))
                chi_dt = chi_tz.strftime(ts)
		message['sensor'] = chi_dt
#		print("Sensor time",chi_dt)

	# GPS data (satellite, altitude)
	if data[0] == '$GPGGA':
		print("****Second data****")
#		print("satellite", data[7])
#		print("altitude", data[9])
		chi_tz = datetime.now(timezone('America/Chicago'))
		chi_dt = chi_tz.strftime(ts)
		message['second'] = chi_dt
#		print("2nd time", chi_dt)

	# GPS data (hdop)
	if data[0] == '$GPGSA':
		print("****Third data****")
#		print("hdop:", data[16])
		chi_tz = datetime.now(timezone('America/Chicago'))
		chi_dt = chi_tz.strftime(ts)
		message['third'] = chi_dt
#		print("3rd time", chi_dt)

	# GPS data (heading)
	if data[0] == '$GPGSV':
		print("****Fourth data****")
#		print("heading:", data[6])
		chi_tz = datetime.now(timezone('America/Chicago'))
		chi_dt = chi_tz.strftime(ts)
		message['fourth'] = chi_dt
#		print("4th time",chi_dt)

	print(message)
	print("----------------------------------")
