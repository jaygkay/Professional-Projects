import serial
import json
import time
gps = serial.Serial("/dev/ttyUSB0", baudrate = 9600)
gpsList = []

ts = time.gmtime()
path = time.strftime("%m%d%y_%H%M%S", ts)

while True:
        line = gps.readline()
        data = line.split(",")

        gpsdata = {}

	if data[0] == '$GPRMC' and data[2] == "A":
		gpsdata['TimeStamp'] = data[1]
		gpsdata['Latitude'] = data[3]
		gpsdata['NS'] = data[4]
		gpsdata['Longitude'] = data[5]
		gpsdata['EW'] = data[6]
		gpsdata['Speed_hp'] = data[7]
		gpsList.append(gpsdata)


	# create json file named by dates and times

	with open("gpsFile_{}.txt".format(path), 'w') as f:
		json.dump(gpsList, f)

	print('Saving json file: {}'.format(path))
