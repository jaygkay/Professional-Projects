# GPS on Raspberry Pi 3  

## Contents  
>### 1. Set parts  
>### 2. Set GPS on RPi3 
>### 3. GPS Data Overview and Python Code  
<br>


## 1. Set parts  

### Parts List  
-Raspberry Pi 3  
-GPS breakout ()
-USB to TLL Serial cable  

### GPS using USB to TTL
Attach the wires from the USB to TLL cable to the GPS breakout:  

| Cable | GPS breakout |   
|:-----------:|:-----------:|  
| Red | VCC |  
| Black | GND |  
| Green | RX |  
| White | TX |  

Once the cable is ready, insert the USB part of the cable into the RPi3. 
<br>

 
## 2. Set GPS on RPi3 

### Installing gpsd (GPS daemon)  
'gpsd' is a service deamon that monitors CPS or AIS receivers attached to a host computer through serial or USB ports.  
 For the more information: http://catb.org/gpsd/.  

Install 'gpsd' and disable the gpsd systemd service.  
```
sudo apt-get install gpsd gpsd-clients python-gps  
sudo systemctl stop gpsd.socket  
sudo systemctl disable gpsd.socket  
```  

The gpsd needs to be started and pointed at the USB device (or UART).  
`sudo killall gpsd`  


USB to TLL use `sudo gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock`  
UART use `sudo gpsd /dev/ttyS0 -F /var/run/gpsd.sock`  


### Checking USB and testing 'gpsd'  
We should select which USB is going to be used by the GPS.  
Following command lists the conndected USB devices (You will see _/dev/ttyUSB0_ typically.):  
`ls /dev/ttyUSB*`  
 
 You can list all USB devices with `sudo lsusb`.  
 
 
 Point 'gpsd' to the USB device.  
 ```
 sudo killall gpsd
 sudo cat /dev/ttyUSB0
 sudo gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock  
  ```  
  
  You can check the lists with `sudo cat /dev/ttyUSB0`.  
  Run `cgps` or `gpsmon` to test and monitor a live-streaming update of GPS data.  
  
  
  

> __Note__  
> If `cgps` does not work, repeat the commands below:
> ```
> sudo systemctl stop gpsd.socket  
> sudo systemctl disable gpsd.socket
> ```  
> then run `sudo gpsd /dev/ttyUSB0 -F /var/run/gpsd.sock`.  
> now rerun `cgps`.    


## 3. GPS Data Overview and Python Code  
```  
sudo pip3 install gps3  
sudo apt-get install python3-microstacknode  
sudo apt-get install python3-serial  
```  


### '$GPxxx' sentence codes and descriptions  
We are using following sentence codes and GPS data:   

| NMEA | DATA |   
|:-----------:|:-----------|  
| $GPRMC | Time, Navigate warning, Latitude, Longitude, Speed(miles/h) |  
| $GPHDT | Heading |  
| $GPVTG | Speed(km/h), Speed(miles/h) |  
| $GPGGA | Altitude |  


> __Note__  
> Format of latitudes and longitudes  
> eg. 4533.35 is 45 degrees and 33.35minutes. '.35' of a minute is 21 seconds.  


More info about NMEA Syntax GPS: http://aprs.gids.nl/nmea/#rmc
