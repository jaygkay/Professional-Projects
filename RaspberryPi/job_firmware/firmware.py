import time
from sense_hat import SenseHat
sensor = SenseHat()

def SenseLight():
        sense = SenseHat()

        X = [255, 0, 0]  # Red
        O = [255, 255, 255]  # White

        download_mark = [
        O, O, O, X, X, O, O, O,
        O, O, O, X, X, O, O, O,
        O, O, O, X, X, O, O, O,
        X, O, O, X, X, O, O, X,
        X, X, O, X, X, O, X, X,
        O, X, X, X, X, X, X, O,
        O, O, X, X, X, X, O, O,
        O, O, O, X, X, O, O, O
        ]

        sense.set_pixels(download_mark)

SenseLight()
time.sleep(10)
sensor.clear()
