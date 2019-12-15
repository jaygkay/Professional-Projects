## Geo Fence (2018.10 ~ 2018.11)

 1. Assign Trip Date 
```python
trip_date = '2018-11-23`
```


 2. `jay.qc_trip_list()` takes one input `trip_date` and outputs a dataframe that includes `camera_id`, `trip_start`, `total_distance` and `location` from the assigned date(from #1) to present.


 3. `jay.redshift_data()` takes two inputs `imei, trip_start` and outputs a dataframe with the corresponding `imei and trip_start` values.
```python
df = jay.redshift_data(imei, trip_start)
```

 4. carvi_trip takes() four inputs `df_redshift` and outputs `firmware_version, UTC Trip_Start/Trip_End, Local Trip_Start/Trip_End, and Basic Trip Information`
```python
jay.trip_info(df)
```
![alt text](https://github.com/jaygkay/projects/blob/master/GeoFencing/png/Screen%20Shot%202018-12-17%20at%204.57.33%20PM.png "ss")

5. sample_geo() takes a parameter of a one-trip-dataframe and prints the information that explains a polygon area and how long the device was staying in the area. This function outputs a dictionary that consists of {'polygon area':['latitude', 'longitude']}
```python
geo_key = sample_geo(df_redshift)
```


 7. You can create the Geo-polygon-area as follows:
```python
garage_zone = [(41.509881, -90.540517),(41.507322, -90.540270),(41.507338, -90.536944),(41.509897, -90.537213)]
centre_station = [(41.507538, -90.522281),(41.505718, -90.521144),(41.507309, -90.517099),(41.508602, -90.518194)]
seventh_st = [(41.470531, -90.528824),(41.470499, -90.527773),(41.498171, -90.527494),(41.498284, -90.528116)]
gas_station = [(41.492974, -90.536377),(41.493472, -90.536361),(41.493458, -90.535827),(41.493026, -90.535848)]
```


 8. geo_key.keys() calls a key 
``` python
geo_key.keys()
```

 9. map_geofence() takes two inputs `geo_key (from #8) and polygon_area`

[Example 1] 

```python
map_geofence(geo_key['garage'], garage_zone)
```

![alt text](https://github.com/jaygkay/projects/blob/master/GeoFencing/png/Screen%20Shot%202018-12-17%20at%205.12.13%20PM.png)


[Example 2]
```python
map_geofence(geo_key['centre'], centre_station)
```
![alt text](https://github.com/jaygkay/projects/blob/master/GeoFencing/png/Screen%20Shot%202018-12-17%20at%205.14.16%20PM.png)

