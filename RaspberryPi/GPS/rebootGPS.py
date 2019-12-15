import os
os.system("sudo killall gpsd")
os.system("sudo cat /dev/ttyUSB0")
os.system("sudo gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock")

os.system("cgps")
