# Time
from datetime import datetime
from pytz import timezone
import os
ts = "%Y%m%d_%H%M%S"

from subprocess import Popen

tzone = datetime.now(timezone('America/Chicago'))
dtime = tzone.strftime(ts)

#videoName = "Rpi_video_{}.mp4".format(dtime)

#avco = Popen(['avconv','-f video4linux2', '-r 25', '-s 1280x960', '-i /dev/video0 /home/pi/aws-iot-device-sdk-python/samples/basicPubSub/Camera/{}'.format(videoName)])

os.system("avconv -f video4linux2 -r 10 -s 1280x960 -i /dev/video0 /home/pi/aws-iot-device-sdk-python/samples/basicPubSub/Camera/Rpi_video_{}.avi".format(dtime))
