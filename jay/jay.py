import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector,time,re,json,googlemaps,os,boto3,sqlalchemy,ast,timezonefinder,gmaps,pytz


from googlemaps import convert
from ast import literal_eval
from pytz import timezone
from datetime import datetime
from tzwhere import tzwhere
from simpledate import SimpleDate
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

class jay(object):
	def __init__(self, ):
		self.timezone = tz
		self.df_redshift = df_redshift
		self.utc_trip_start = utc_trip_start
		self.utc_year = utc_year
		self.utc_hour = utc_hour
		self.utc_trip_start = utc_trip_start
		self.prefix_carvi_all = prefix_carvi_all
		self.prefix_carvi_event = prefix_carvi_all
		self.prefix_carvi_end = prefix_carvi_end
		self.prefix_carvi_request = prefix_carvi_request
		self.prefix_carvi_crash = prefix_carvi_crash
		self.prefix_aws_connected = prefix_aws_connected
		self.prefix_aws_disconnected = prefix_aws_disconnected
		return self
	
	def google_map(df):

		location_lst = [tuple(ast.literal_eval(loc)) for loc in df.location]
		cnt_lst = []
	    
		for i in location_lst:
			cnt_lst.append(1)
	    
		new_loc = pd.DataFrame({'location':location_lst,'cnt':cnt_lst})
		new_loc = new_loc[new_loc['location'] != (0.0000,0.0000)]    
	    
		if len(new_loc) == 0:
			fig = print(colored("\nGPS LOCATIONS IS MISSING", "red", attrs = ['bold']))
		else:
			gmaps.configure(api_key='---')
			fig = gmaps.figure()
			###############################
			percent = len(new_loc.location)

			start_point = new_loc.location.iloc[0]
			end_point = new_loc.location.iloc[-1]

			trips = [
			    {'trip':'trip start', 'location' : start_point},
			    {'trip':'trip end', 'location' : end_point}]
			trip_locations = [trip['location'] for trip in trips]
			info_box_template = """
			<dl>
			<dt><b><font color="red">Trip</font></b></dt>
			<dd><center><font color="blue">{trip}</font></center></dd>
			</dl>
			"""
			trip_info = [info_box_template.format(**trip) for trip in trips]
			marker_layer = gmaps.marker_layer(trip_locations, info_box_content=trip_info)

			markers = gmaps.marker_layer([start_point,end_point])
			fig.add_layer(marker_layer)

			###############################
			fig.add_layer(gmaps.heatmap_layer(new_loc.location, weights = new_loc['cnt'],
	                                          max_intensity = 1, point_radius = 3, opacity = 0.7))
		return fig


	def redshift_data(imei, trip_start):
		'''
   		This function takes 'imei number' and 'trip_start'
   		This function returns a dataframe from redshift that contains all feature with trip_start as a key

   		ex) redshift_data(str_imei, str_trip_start)
   		'''
		engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
		trip_query_df = pd.read_sql_query("SELECT * FROM carvi_normal_data WHERE camera_id = '"+imei+"' AND trip_start = '"+trip_start+ "';", engine)
		trip_query_df = trip_query_df.sort_values(['time_stamp'])
		trip_query_df = trip_query_df.reset_index()

		return trip_query_df

	def poc_all(trip_start, trip_end):
		'''
   		This function takes 'imei number' and 'trip_start'
   		This function returns a dataframe from redshift that contains all feature with trip_start as a key

   		ex) redshift_data(str_imei, str_trip_start)
   		'''
		engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
		trip_query_df = pd.read_sql_query("\
			SELECT  camera_id,trip_start,time_stamp,collision_distance, \
			distance,hdop,satellite,speed,location, prewarn,continuity, \
			bias,bias_ls,situation,reaction,reduction_speed,direction, \
			front_distance,front_reduction_speed,front_speed,front_reason,\
			ttc,left_lane,right_lane,speed_skor,focus_skor,guard_skor,lat,lon,event,heading\
			FROM carvi_normal_data \
			WHERE camera_id = ANY (ARRAY['861107036575935', '861107036571645', '861107036576461', \
			'861107036567072','861107036571231','861107036569862','861107036568377','861107036574755',\
			'861107036570373','861107036575927','861107036575463', '861107036571272', '861107036575059',\
			'861107036574912','861107036572064','861107036570217','861107036569797','861107036569953',\
			'861107036571868','861107036575521','861107036566751','861107036575364','861107036575596'])\
			AND trip_start >= '"+trip_start+ "'\
			AND trip_start < '"+trip_end+ "';", engine)

		trip_query_df = trip_query_df.sort_values(['time_stamp'])
		trip_query_df = trip_query_df.reset_index()

		return trip_query_df



	def start_end(camera_id,trip_start,trip_end):
	    engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
	    trip_query = pd.read_sql_query("\
	      SELECT camera_id, COUNT(*) as data_size,\
	      MAX(distance) as distance,\
	      trip_start,\
	      MAX(CASE WHEN st.last_row=1 THEN st.time_stamp END) as trip_end,\
	      MAX(CASE WHEN st.first_row=1 THEN st.event END) as start_event,\
	      MAX(CASE WHEN st.last_row=1 THEN st.event END) as end_event,\
	    MAX(CASE WHEN st.first_row=1 THEN st.location END) as location,\
	      MAX(CASE WHEN st.last_row=2 THEN st.location END) as end_loc,\
	      count(case when speed = 0.0 then st.speed END) as idling_cnt\
	      FROM (SELECT camera_id, trip_start, time_stamp,distance, event, \
	      location, speed,version, situation,\
	        ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp ASC) AS first_row, \
	        ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp DESC) AS last_row \
	    FROM carvi_normal_data\
	      WHERE camera_id = '"+camera_id+"'\
	      AND lat > 0\
	      AND trip_start >= '"+trip_start+"' \
	      AND trip_start < '"+trip_end+"') st\
	      GROUP BY camera_id, trip_start\
	      ORDER BY trip_start;", engine)  
	    #trip_query = trip_query.sort_values(['trip_start'])
	    trip_query = trip_query.reset_index()
	    trip_query = trip_query.drop('index',axis = 1)
	    #trip_query = trip_query[trip_query.distance>0.001]
	    return trip_query
	    
	def multiple_data(imei, trip_start, trip_end):
		'''
   		This function takes 'imei number' and 'trip_start'
   		This function returns a dataframe from redshift that contains all feature with trip_start as a key

   		ex) redshift_data(str_imei, str_trip_start)
   		'''
		engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
		trip_query_df = pd.read_sql_query("SELECT * FROM carvi_normal_data WHERE camera_id in "+str(tuple(imei))+" \
										   AND trip_start >= '"+trip_start+ "'\
										   AND trip_start < '"+trip_end+ "';", engine)

		trip_query_df = trip_query_df.sort_values(['time_stamp'])
		trip_query_df = trip_query_df.reset_index()

		return trip_query_df  

	def trip_list(imei, trip_start):
		engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev') 
		trip_start_query = pd.read_sql_query("\
			SELECT subTbl.camera_id, subTbl.trip_start,subTbl.location, \
				   MAX(distance) AS distance, min(version) AS firmware \
			FROM ( SELECT \
				camera_id, trip_start,location, \
				row_number() OVER(partition by trip_start order by location desc) AS roworder\
				FROM carvi_normal_data WHERE camera_id = '"+imei+"' \
				AND trip_start >= '"+trip_start+"' \
				AND location <> '0.0000,0.0000'\
				) subTbl, carvi_normal_data mainTbl\
			WHERE roworder = 1 AND subTbl.camera_id = mainTbl.camera_id\
				AND subTbl.trip_start = mainTbl.trip_start\
			GROUP BY subTbl.camera_id, subTbl.trip_start,subTbl.location\
			ORDER BY trip_start;", engine)
		trip_start_query = trip_start_query.sort_values(['trip_start'])#, ascending = True)
		trip_start_query = trip_start_query.reset_index()
		trip_start_query = trip_start_query.drop('index',axis = 1)
		return trip_start_query

	def poc_list(imei, trip_start,trip_end):
		engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev') 
		trip_start_query = pd.read_sql_query("\
			SELECT subTbl.camera_id, subTbl.trip_start,subTbl.location, \
				   MAX(distance) AS distance, min(version) AS firmware \
			FROM ( SELECT \
				camera_id, trip_start,location, \
				row_number() OVER(partition by trip_start order by location desc) AS roworder\
				FROM carvi_normal_data WHERE camera_id = '"+imei+"' \
				AND trip_start >= '"+trip_start+"' \
				AND trip_start < '"+trip_end+"' \
				AND location <> '0.0000,0.0000'\
				) subTbl, carvi_normal_data mainTbl\
			WHERE roworder = 1 AND subTbl.camera_id = mainTbl.camera_id\
				AND subTbl.trip_start = mainTbl.trip_start\
			GROUP BY subTbl.camera_id, subTbl.trip_start,subTbl.location\
			ORDER BY trip_start;", engine)
		trip_start_query = trip_start_query.sort_values(['trip_start'])#, ascending = True)
		trip_start_query = trip_start_query.reset_index()
		trip_start_query = trip_start_query.drop('index',axis = 1)
		return trip_start_query

	def multi_redshift_data(imei, trip_start, trip_end):
	    '''
	    This function takes 'imei number' and 'trip_start'
	    This function returns a dataframe from redshift that contains all feature with trip_start as a key

	    ex) redshift_data(str_imei, str_trip_start)
	    '''
	    engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
	    trip_query_df = pd.read_sql_query("SELECT * FROM carvi_normal_data \
	    WHERE camera_id = '"+imei+"' \
	    AND trip_start >= '"+trip_start+ "'\
	    AND trip_start < '"+trip_end+ "'\
	    order by camera_id, trip_start;", engine)
	    trip_query_df = trip_query_df.sort_values(['time_stamp'])
	    trip_query_df = trip_query_df.reset_index()
	    
	    return trip_query_df 


	def local_timezone(index, df):
		
	    trip_start = df['trip_start'].iloc[index]
	    gps = df['location'].iloc[index]
	    tz = tzwhere.tzwhere()
	    lat = float(gps.split(',')[0])
	    lng = float(gps.split(',')[1])
	    timezone = tz.tzNameAt(lat, lng)
	    return timezone

	def time_zone(time_zone):
	    tz = tzwhere.tzwhere()
	    lat = float(time_zone.split(',')[0])
	    lng = float(time_zone.split(',')[1])
	    timezone = tz.tzNameAt(lat, lng)
	    return timezone

	def convert_utc_trip_start(trip_start,tz):
	    utc_trip_start = SimpleDate(trip_start, tz = tz).utc
	    utc_trip_start = str(utc_trip_start)
	    return utc_trip_start


	def local_utc_time(event_start, trip_start, start, utc_trip_start,tz):
	    start_split = utc_trip_start.split('-')
	    utc_year = start_split[0]+'/'+start_split[1]+'/'+start_split[2][0:2]
	    utc_hour = start_split[2][3:5]
	    print(colored(start,"green",attrs = ['bold']),
	    	  colored("with","green",attrs = ['bold']),
	    	  colored(event_start,"red", attrs = ['bold']))
	    print(colored('Local '+start,'blue',attrs = ['bold']),':',
	          colored(tz,'red', attrs = ['bold']), 
	          colored(trip_start, 'red'))
	    print(colored('S3_object_key '+str(start),'blue',attrs = ['bold']),':',
	          colored('UTC','red',attrs = ['bold']),
	          colored(utc_trip_start, 'red'))
	    return utc_year, utc_hour, utc_trip_start

	def trip_info(df_redshift):
	    event = df_redshift.event.value_counts()
	    df_redshift['time_stamp'] = pd.to_datetime(df_redshift['time_stamp'])
	    print(colored("\nData Shape :", "blue", attrs = ['bold']), colored(df_redshift. shape,"red"))
	    
	    dist = np.max(df_redshift['distance'])   
	    print(colored("Trip Distance:", "blue", attrs = ['bold']),
	          colored(dist, "red"),
	          colored("km", "red"))
	    time_diff = df_redshift['time_stamp'].iloc[-1] - df_redshift['time_stamp'].iloc[0]
	    total_seconds = int(time_diff.total_seconds())
	    hours, remainder = divmod(total_seconds,60*60)
	    minutes, seconds = divmod(remainder,60)
	    print(colored("Trip Duration:", "blue", attrs = ['bold']),
	          colored('{} Hours {} Minutes {} Seconds'.format(hours,minutes,seconds), 'red'))

	def time_gap(df): 
		df['time_stamp'] = pd.to_datetime(df['time_stamp'])
		time_gap = df['time_stamp'].diff()
		gap = time_gap[time_gap>'0 days 00:00:02']

		if len(gap) < 1:
			print('No time gap > 2 seconds')
		else:
			print(colored('Time gap > 2 seconds are detected','red'))
			for i, j in zip(gap.index, gap):
				print(colored('------------------------------------------------------------------------------','blue'))
				print(df[['time_stamp','event','location','speed','satellite']].iloc[i-1:i+2])
		print(colored('------------------------------------------------------------------------------','blue'))

	def aws_s3_prefix(bucket, aws_start_year, imei, firmware):
	    prefix_carvi_all = aws_start_year+'/CarVi/test/'+imei+'/all/'+firmware+'/'
	    prefix_carvi_event = aws_start_year+'/CarVi/test/'+imei+'/event/'+firmware+'/'
	    prefix_carvi_end = aws_start_year+'/CarVi/test/'+imei+'/end/'+firmware+'/'
	    prefix_carvi_request = aws_start_year+'/CarVi/test/'+imei+'/request/'+firmware+'/'
	    prefix_carvi_crash = aws_start_year+'/CarVi/test/'+imei+'/crash/'+firmware+'/'
	    prefix_aws_connected = aws_start_year+'/$aws/events/presence/connected/'+imei+'/'
	    prefix_aws_disconnected = aws_start_year+'/$aws/events/presence/disconnected/'+imei+'/'
	   # print(prefix_carvi_all)
	    return prefix_carvi_all, prefix_carvi_event, prefix_carvi_end, prefix_carvi_request, prefix_carvi_crash,prefix_aws_connected,prefix_aws_disconnected
	def utc_conv(unixtime):
	    unixtime = str(unixtime)
	    if len(unixtime) <= 13:
	        zero_cnt = 13 - len(unixtime)
	        unixtime = unixtime + '0' * zero_cnt
	        
	    unix_timestamp = float(unixtime)/1000
	    utctz = pytz.timezone(zone)
	    utc_dt = str(utctz.fromutc(datetime.utcfromtimestamp(unix_timestamp).replace(tzinfo=utctz)))
	    return utc_dt[:-9]

	def dump_s3_data(env, bucket, imei, utc_start_year,utc_start_hour, utc_end_hour,firmware):
		prefix_carvi_all, prefix_carvi_event, prefix_carvi_end, prefix_carvi_request, prefix_carvi_crash,prefix_aws_connected,prefix_aws_disconnected = jay.aws_s3_prefix(bucket, utc_start_year, imei, firmware)
		env= boto3.session.Session(profile_name=str(env))
		s3 = env.client('s3')
		paginator = s3.get_paginator('list_objects')
	    #for page in page_iter:
		if utc_start_hour != utc_end_hour:
	        #print('NOT SAME')
			aws_hours = []
			for i in range(int(utc_start_hour), int(utc_end_hour)+1):
				aws_hours.append(str(i).zfill(2))

			prefix_lst = []
			prefix_lst.append(prefix_carvi_event + utc_start_hour )
			for i in aws_hours:
				prefix_lst.append(prefix_carvi_all+str(i))
	            #### prefix_carvi_crash
			prefix_lst.append(prefix_carvi_end + utc_end_hour)
	#         prefix_lst
		else:
	        #print('SAME')
			prefix_lst = [prefix_carvi_event + utc_start_hour ,
	                      prefix_carvi_all + utc_start_hour,
	                      prefix_carvi_crash + utc_start_hour,
	                      prefix_carvi_end + utc_start_hour]

		key_lst = []
	#    print(prefix_lst)
	#     print(colored("S3 Bucket for This","blue"), colored(imei, "grey"))
		print("\t",colored("reading keys from S3 buckets", "green", attrs = ['bold']))
		for prefix_i in prefix_lst:
	#         print(prefix_i)
			params = {'Bucket' : bucket, 'Prefix' : prefix_i}
			page_iter = paginator.paginate(Bucket = params['Bucket'],
	                                       Prefix = params['Prefix'])
			for page in page_iter:
				try:
					#print(prefix_i)
					key_lst += [(x['Key']) for x in page['Contents']]
				except KeyError:
					error = prefix_i.split('/')
	                #print(error)
					print("\t",colored(error[6],"grey"),
							   colored("message no exists in ", "red"),
							   colored(prefix_i, "red"))
					continue
				else:
					print("\t",colored(prefix_i, "green",attrs = ['bold']))
		print("\t",colored("Dumping S3 Data to a DataFrame", "green"))
		df2 = pd.DataFrame()
		for i in key_lst:
	        #print(i)
			response = s3.get_object(Bucket = bucket, Key = i)
			result = response['Body'].read()
	        #print("********", result)
			try:
				trip_json = json.loads(result)
				if isinstance(trip_json, dict):
	#                 if trip_json['trip_start'] == trip_start: 
					df2 = df2.append(trip_json,ignore_index=True )
				else:
					for trip in trip_json:
	#                     if trip['trip_start'] == trip_start: 
						df2 = df2.append(trip,ignore_index=True )
	            
			except json.JSONDecodeError:
				error  = trip_json
				print(error)
	        
	    #df2 = df2.sort_values(['time_stamp'])
		df2 = df2.reset_index()
		return df2

	def print_devices():

	    IMEI_list = pd.read_excel('Carvi_IMEI.xlsx')
	    print(IMEI_list.Company.value_counts())
	
	def load_columns():
	    file = pd.read_excel('poc_col.xlsx')
	    return file   

	def load_devices(company):
  
	    IMEI_list = pd.read_excel('Carvi_IMEI.xlsx')
	    camera_ids_lst = IMEI_list.IMEI[IMEI_list.Company == company].tolist()
	    camera_ids = camera_ids_lst #','.join(str(i) for i in camera_ids_lst)

	    return camera_ids

	def company_trip(company_imei,trip_start,trip_end):
	    engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
	    trip_query = pd.read_sql_query("\
          SELECT camera_id, trip_start,\
          MAX(CASE WHEN st.last_row=1 THEN st.time_stamp END) as trip_end,\
          MAX(CASE WHEN st.first_row=1 THEN st.event END) as start_event,\
          MAX(CASE WHEN st.last_row=1 THEN st.event END) as end_event,\
          MAX(CASE WHEN st.first_row=2 THEN st.distance END) as start_dist,\
          MAX(CASE WHEN st.last_row=2 THEN st.distance END) as end_dist,\
          MAX(location) AS location,\
          COUNT(*) as data_size,\
          count(case when location = '0.00000,0.00000' then st.location END) as zero_loc,\
          count(case when speed = 0.0 then st.speed END) as idling_cnt,\
          count(case when event = 'collision' then st.event END) as fcw,\
          count(case when event = 'departure' then st.event END) as ldw,\
          count(case when situation = 'accel' then st.situation END) as accel,\
          count(case when situation = 'brake' then st.situation END) as brake,\
          count(case when situation = 'stop' then st.situation END) as stop,\
          count(case when situation = 'front' then st.situation END) as front,\
          count(case when event = 'sudden' then st.event END) as sudden,\
          MAX(CASE WHEN st.first_row=1 THEN st.version END) as version\
          FROM (SELECT camera_id, trip_start, time_stamp,distance, event, \
          location, speed,version, situation,\
            ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp ASC) AS first_row, \
            ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp DESC) AS last_row \
        FROM carvi_normal_data\
          WHERE camera_id in "+str(tuple(company_imei))+"\
          AND trip_start >= '"+trip_start+"' \
          AND trip_start < '"+trip_end+"') st\
          GROUP BY camera_id, trip_start\
          ORDER BY trip_start;", engine)  
	    #trip_query = trip_query.sort_values(['trip_start'])
	    trip_query = trip_query.reset_index()
	    trip_query = trip_query.drop('index',axis = 1)
	    #trip_query = trip_query[trip_query.distance>0.001]
	    return trip_query

	def company_single(company_imei,trip_start,trip_end):
	    engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
	    trip_query = pd.read_sql_query("\
          SELECT camera_id, trip_start,\
          MAX(CASE WHEN st.last_row=1 THEN st.time_stamp END) as trip_end,\
          MAX(CASE WHEN st.first_row=1 THEN st.event END) as start_event,\
          MAX(CASE WHEN st.last_row=1 THEN st.event END) as end_event,\
          MAX(CASE WHEN st.first_row=2 THEN st.distance END) as start_dist,\
          MAX(CASE WHEN st.last_row=2 THEN st.distance END) as end_dist,\
          MAX(location) AS location,\
          COUNT(*) as data_size,\
          count(case when location = '0.00000,0.00000' then st.location END) as zero_loc,\
          count(case when speed = 0.0 then st.speed END) as idling_cnt,\
          count(case when event = 'collision' then st.event END) as fcw,\
          count(case when event = 'departure' then st.event END) as ldw,\
          count(case when situation = 'accel' then st.situation END) as accel,\
          count(case when situation = 'brake' then st.situation END) as brake,\
          count(case when situation = 'stop' then st.situation END) as stop,\
          count(case when situation = 'front' then st.situation END) as front,\
          count(case when event = 'sudden' then st.event END) as sudden,\
          MAX(CASE WHEN st.first_row=1 THEN st.version END) as version\
          FROM (SELECT camera_id, trip_start, time_stamp,distance, event, \
          location, speed,version, situation,\
            ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp ASC) AS first_row, \
            ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp DESC) AS last_row \
        FROM carvi_normal_data\
          WHERE camera_id = '"+str(company_imei)+"'\
          AND trip_start >= '"+trip_start+"' \
          AND trip_start < '"+trip_end+"') st\
          GROUP BY camera_id, trip_start\
          ORDER BY trip_start;", engine)  
	    #trip_query = trip_query.sort_values(['trip_start'])
	    trip_query = trip_query.reset_index()
	    trip_query = trip_query.drop('index',axis = 1)
	    #trip_query = trip_query[trip_query.distance>0.001]
	    return trip_query

	def single_trip(company_imei,trip_start,trip_end):
	    engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
	    trip_query = pd.read_sql_query("\
          SELECT camera_id, trip_start,\
          MAX(CASE WHEN st.last_row=1 THEN st.time_stamp END) as trip_end,\
          MAX(CASE WHEN st.first_row=1 THEN st.event END) as start_event,\
          MAX(CASE WHEN st.last_row=1 THEN st.event END) as end_event,\
          MAX(CASE WHEN st.first_row=2 THEN st.distance END) as start_dist,\
          MAX(CASE WHEN st.last_row=2 THEN st.distance END) as end_dist,\
          MAX(location) AS location,\
          COUNT(*) as data_size,\
          count(case when location = '0.00000,0.00000' then st.location END) as zero_loc,\
          MAX(CASE WHEN st.first_row=1 THEN st.version END) as version\
          FROM (SELECT camera_id, trip_start, time_stamp,distance, event, \
          location, speed,version, situation,\
            ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp ASC) AS first_row, \
            ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp DESC) AS last_row \
        FROM carvi_normal_data\
          WHERE camera_id = '"+company_imei+"'\
          AND trip_start >= '"+trip_start+"' \
          AND trip_start < '"+trip_end+"') st\
          GROUP BY camera_id, trip_start\
          ORDER BY trip_start;", engine)  
	    #trip_query = trip_query.sort_values(['trip_start'])
	    trip_query = trip_query.reset_index()
	    trip_query = trip_query.drop('index',axis = 1)
	    #trip_query = trip_query[trip_query.distance>0.001]
	    return trip_query

	def pr_trip(trip_start,trip_end):
	    engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
	    trip_query = pd.read_sql_query("\
          SELECT camera_id, trip_start,\
          MAX(CASE WHEN st.last_row=2 THEN st.distance END) as mileage,\
          COUNT(*) as data_size,\
          MAX(location) AS location,\
          count(case when speed = 0.0 then st.speed END) as idling_cnt,\
          count(case when event = 'collision' then st.event END) as coll_cnt,\
          count(case when event = 'departure' then st.event END) as departure,\
          max(case when collision_distance > 0 then st.collision_distance END) as coll_dist,\
          count(case when event= 'front' then st.event END) as front,\
          count(case when event= 'sudden' then st.event END) as sudden,\
          count(case when situation= 'accel' then st.situation END) as accel,\
          count(case when situation= 'brake' then st.situation END) as brake,\
          count(case when situation= 'stop' then st.situation END) as stop,\
          count(case when situation= 'reduction' then st.situation END) as reduction,\
          count(case when ttc>0 then st.ttc END) as ttc\
          FROM (SELECT camera_id, trip_start, time_stamp, distance, event,\
            speed, ttc, collision_distance ,situation,location,\
            ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp ASC) AS first_row, \
            ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp DESC) AS last_row \
        FROM carvi_normal_data\
          WHERE trip_start >= '"+trip_start+"' \
          AND trip_start < '"+trip_end+"') st\
          GROUP BY camera_id, trip_start\
          ORDER BY trip_start;", engine)  
	    #trip_query = trip_query.sort_values(['trip_start'])
	    trip_query = trip_query.reset_index()
	    trip_query = trip_query.drop('index',axis = 1)
	    #trip_query = trip_query[trip_query.distance>0.001]
	    return trip_query


	def pr_trip2(trip_start,trip_end):
	    engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
	    trip_query = pd.read_sql_query("\
              SELECT camera_id, trip_start, time_stamp, speed, event, situation, collision_distance,ttc\
            FROM carvi_normal_data\
              WHERE \
              AND trip_start >= '"+trip_start+"' \
              AND trip_start < '"+trip_end+"'\
              order BY camera_id, trip_start;", engine)  
	    #trip_query = trip_query.sort_values(['trip_start'])
	    trip_query = trip_query.reset_index()
	    trip_query = trip_query.drop('index',axis = 1)
	    #trip_query = trip_query[trip_query.distance>0.001]
	    return trip_query

	def exp_data_size(df):
	    df.trip_start = pd.to_datetime(df.trip_start)
	    df.trip_end = pd.to_datetime(df.trip_end)
	    df['time_diff'] = round(pd.to_timedelta(df.trip_end - df.trip_start).dt.total_seconds(),0)
	    df.fillna(0)

	    exp1 = []
	    exp2 = []
	    for i in df.time_diff:
	        val1 = int(i - i*(.0167))
	        val2 = int(i + i*(.0167))
	        exp1.append(val1)
	        exp2.append(val2)
	    df['min_expt_size'] = exp1
	    df['max_expt_size'] = exp2

	    thus = []
	    for i,j,k in zip(df.data_size, df.min_expt_size, df.max_expt_size):
	        val = i in range(j,k+1)
	        thus.append(val)

	    df['validation'] = thus
	    return df
	def validation(company, trip_start, trip_end):
	    # cols: column names for a dataframe that will be created next cell.
	    cols =  'company', 'trip_date','trip_count', 'total_trip_sec', 'total_data_row','data_quality','missing_gps','valid_gps', 'idling_cnt','fcw','ldw','sudden','accel','brake','stop',
	    # vals: the values that seen in the columns will be appended by company
	    vals = []
	    
	    for comp in company:
	        # company_imei assigns a company name as a parameter
	        company_imei = jay.load_devices(comp)
	        # jay.company_trip returns the dataframe that contains the query result in the class()
	        company_df = jay.company_trip(company_imei,trip_start, trip_end)
	        # jay.exp_data_size() returns the dataframe that contains the expected datasize with confidence intervals with 1.67%
	        #exp_size = jay.exp_data_size(company_df)
	        company_df.trip_start = pd.to_datetime(company_df.trip_start)
	        company_df.trip_end = pd.to_datetime(company_df.trip_end)
	        company_df['time_diff'] = round(pd.to_timedelta(company_df.trip_end - company_df.trip_start).dt.total_seconds(),0)



	        # these are the vals that will be appneded to the dataframe
	        data_size = len(company_df)
	        if data_size == 0:
	        	print("No Trip for",comp)
	        	pass
	        else:
	        	print('validating',comp)
		        total_trip_sec = company_df.time_diff.sum()
		        total_data_row = company_df.data_size.sum()
		        missing_gps = company_df.zero_loc.sum()
		        idle_cnt = company_df.idling_cnt.sum()
		        trip_date = trip_start
		        fcw = company_df.fcw.sum()
		        ldw = company_df.ldw.sum()
		        sudden = company_df.sudden.sum()
		        accel = company_df.accel.sum()
		        brake = company_df.brake.sum()
		        stop = company_df.stop.sum()
		        front = company_df.front.sum()

	        # if total_data_row == 0:
	        #     valid_gps = 0
	        #     data_quality = 0
	        # else:
		        valid_gps = round((1-(missing_gps/total_data_row)) * 100, 2)
		        data_quality = round((total_data_row/total_trip_sec) * 100,2)
	        	val = comp,trip_date,data_size,total_trip_sec,total_data_row,data_quality,missing_gps,valid_gps,idle_cnt,fcw,ldw,sudden,accel,brake,stop

	        # append the vals above
	        	vals.append(val)

	    df = pd.DataFrame(vals, columns = cols)
	    df.replace(False, 0, inplace=True)
	    return df

	def validation_api(company, trip_start, trip_end):
	    # cols: column names for a dataframe that will be created next cell.
	    cols =  'company', 'trip_date','trip_count', 'total_trip_sec', 'total_data_row','data_quality','missing_gps','valid_gps'
	    # vals: the values that seen in the columns will be appended by company
	    vals = []
	    
	    for comp in company:
	        # company_imei assigns a company name as a parameter
	        company_imei = jay.load_devices(comp)
	        if len(company_imei)>1:
	        # jay.company_trip returns the dataframe that contains the query result in the class()
	        	company_df = jay.company_trip(company_imei,trip_start, trip_end)
	        else:
	        	company_df = jay.company_single(company_imei[0],trip_start, trip_end)
	        # jay.exp_data_size() returns the dataframe that contains the expected datasize with confidence intervals with 1.67%
	        #exp_size = jay.exp_data_size(company_df)
	        company_df.trip_start = pd.to_datetime(company_df.trip_start)
	        company_df.trip_end = pd.to_datetime(company_df.trip_end)
	        company_df['time_diff'] = round(pd.to_timedelta(company_df.trip_end - company_df.trip_start).dt.total_seconds(),0)

	        # these are the vals that will be appneded to the dataframe
	        data_size = len(company_df)
	        if data_size == 0:
	        	print("No Trip for",comp)
	        	pass
	        else:
	        	print('validating',comp)
		        total_trip_sec = company_df.time_diff.sum()
		        total_data_row = company_df.data_size.sum()
		        missing_gps = company_df.zero_loc.sum()
		        idle_cnt = company_df.idling_cnt.sum()
		        trip_date = str(trip_start.split(' ')[0])
		  
		        valid_gps = round((1-(missing_gps/total_data_row)) * 100, 2)
		        data_quality = round((total_data_row/total_trip_sec) * 100,2)
	        	val = comp,trip_date,data_size,total_trip_sec,total_data_row,data_quality,missing_gps,valid_gps

	      
	        	vals.append(val)

	    df = pd.DataFrame(vals, columns = cols)
	    df.replace(False, 0, inplace=True)
	    return df


	def qc_api(company, trip_start, trip_end):
	    # cols: column names for a dataframe that will be created next cell.
		cols =  'company', 'trip_date','trip_count', 'total_trip_sec', 'total_data_row','data_quality','missing_gps','valid_gps'
	    # vals: the values that seen in the columns will be appended by company
		vals = []
	    
		for comp in company:
	        # company_imei assigns a company name as a parameter
			company_imei = jay.load_devices(comp)
			if len(company_imei)>1:
	        # jay.company_trip returns the dataframe that contains the query result in the class()
				company_df = jay.company_trip(company_imei,trip_start, trip_end)
			else:
				company_df = jay.company_single(company_imei[0],trip_start, trip_end)
	        # jay.exp_data_size() returns the dataframe that contains the expected datasize with confidence intervals with 1.67%
	        #exp_size = jay.exp_data_size(company_df)
			company_df.trip_start = pd.to_datetime(company_df.trip_start)
			company_df.trip_end = pd.to_datetime(company_df.trip_end)
			company_df['time_diff'] = round(pd.to_timedelta(company_df.trip_end - company_df.trip_start).dt.total_seconds(),0)
			# version = company_df.version.iloc[0]
	        # these are the vals that will be appneded to the dataframe
			data_size = len(company_df)
			if data_size == 0:
				print("No Trip for",comp)
				pass
			else:
				print('validating',comp)
				total_trip_sec = company_df.time_diff.sum()
				total_data_row = company_df.data_size.sum()
				missing_gps = company_df.zero_loc.sum()
				idle_cnt = company_df.idling_cnt.sum()
				trip_date = str(trip_start.split(' ')[0])
				
				valid_gps = round((1-(missing_gps/total_data_row)) * 100, 2)
				data_quality = round((total_data_row/total_trip_sec) * 100,2)
				val = comp,trip_date,data_size,total_trip_sec,total_data_row,data_quality,missing_gps,valid_gps
	      
				vals.append(val)

		df = pd.DataFrame(vals, columns = cols)
		df.replace(False, 0, inplace=True)
		return df

	def single_validation(company, trip_start, trip_end):
	    # cols: column names for a dataframe that will be created next cell.
		cols =  'company', 'trip_date','trip_count', 'total_trip_sec', 'total_data_row','data_quality','missing_gps','valid_gps'
		# vals: the values that seen in the columns will be appended by company
		vals = []
		date_lst = []
		for each in company:
			print(each)
			comp_lst = jay.load_devices(each)
			for comp in comp_lst:
			    # company_imei assigns a company name as a parameter
			    #company_imei = jay.load_devices(comp)
			    # jay.company_trip returns the dataframe that contains the query result in the class()
				company_df = jay.single_trip(str(comp),trip_start, trip_end)
			    # jay.exp_data_size() returns the dataframe that contains the expected datasize with confidence intervals with 1.67%
			    #exp_size = jay.exp_data_size(company_df)
				#print(company_df.trip_start,'!!!!!!!!!!!!!!!!!!!!!')
				company_df.trip_start = pd.to_datetime(company_df.trip_start)
				company_df.trip_end = pd.to_datetime(company_df.trip_end)
				company_df['time_diff'] = round(pd.to_timedelta(company_df.trip_end - company_df.trip_start).dt.total_seconds(),0)
		        # these are the vals that will be appneded to the dataframe
				data_size = len(company_df)
				if data_size == 0:
					print("Validation API: No Trip for",comp)
					val = comp,0,0,0,0,0,0,0
					vals.append(val)
				else:
					print('Validation API: validating',comp)
					total_trip_sec = company_df.time_diff.sum()
					total_data_row = company_df.data_size.sum()
					missing_gps = company_df.zero_loc.sum()
					date_lst = []
					for i in range(0, len(company_df)):
						# print(str(company_df.trip_start))
						#trip_date = str(company_df.trip_start[i])[:-3]
						#print('!!!!!!!!!',str(company_df.trip_start[i]).split('.'))
						ymt_hms = str(company_df.trip_start[i]).split('.')[0]
						if len(str(company_df.trip_start[i]).split('.')) == 1:
							trip_date = ymt_hms
							#print('1',trip_date)
							
						else: 
							if len(str(company_df.trip_start[i]).split('.')) == 2:
								if len(str(company_df.trip_start[i]).split('.')[1]) == 6:
									millie = str(company_df.trip_start[i]).split('.')[1][:3]
								else:
									millie = str(company_df.trip_start[i]).split('.')[1]
								trip_date = ymt_hms+'.'+millie
								#print('2',trip_date)

						#print(ymt_hms+'.'+millie,str(company_df.trip_start[i])[:-3])
						#trip_date = ymt_hms+'.'+millie
						date_lst.append(trip_date)
						#print(trip_date)

					valid_gps = round((1-(missing_gps/total_data_row)) * 100, 2)
					if total_trip_sec == 0:
						data_quality = 0
					else:
						data_quality = round((total_data_row/total_trip_sec) * 100,2)
					val = comp,date_lst,data_size,total_trip_sec,total_data_row,data_quality,missing_gps,valid_gps
					vals.append(val)

		df = pd.DataFrame(vals, columns = cols)
		df.replace(False, 0, inplace=True)
		return df
    
	def skor_validation(company, trip_start, trip_end):
	    # cols: column names for a dataframe that will be created next cell.
		cols =  'company', 'trip_date','trip_count', 'total_trip_sec', 'total_data_row','data_quality','missing_gps','valid_gps'
		# vals: the values that seen in the columns will be appended by company
		vals = []
		date_lst = []
		for each in company:
			print(each)
			comp_lst = jay.load_devices(each)
			for comp in comp_lst:
			    # company_imei assigns a company name as a parameter
			    #company_imei = jay.load_devices(comp)
			    # jay.company_trip returns the dataframe that contains the query result in the class()
				company_df = jay.single_trip(str(comp),trip_start, trip_end)
			    # jay.exp_data_size() returns the dataframe that contains the expected datasize with confidence intervals with 1.67%
			    #exp_size = jay.exp_data_size(company_df)
				#print(company_df.trip_start,'!!!!!!!!!!!!!!!!!!!!!')
				company_df.trip_start = pd.to_datetime(company_df.trip_start)
				company_df.trip_end = pd.to_datetime(company_df.trip_end)
				company_df['time_diff'] = round(pd.to_timedelta(company_df.trip_end - company_df.trip_start).dt.total_seconds(),0)
				total_trip_sec = company_df.time_diff.sum()
		        # these are the vals that will be appneded to the dataframe
				data_size = len(company_df)
				if total_trip_sec == 0:
					break
				else:
					print('Validation API: validating',comp)
					total_trip_sec = company_df.time_diff.sum()
					total_data_row = company_df.data_size.sum()
					missing_gps = company_df.zero_loc.sum()
					date_lst = []
					for i in range(0, len(company_df)):
						# print(str(company_df.trip_start))
						#trip_date = str(company_df.trip_start[i])[:-3]
						#print('!!!!!!!!!',str(company_df.trip_start[i]).split('.'))
						ymt_hms = str(company_df.trip_start[i]).split('.')[0]
						if len(str(company_df.trip_start[i]).split('.')) == 1:
							trip_date = ymt_hms
							#print('1',trip_date)
							
						else: 
							if len(str(company_df.trip_start[i]).split('.')) == 2:
								if len(str(company_df.trip_start[i]).split('.')[1]) == 6:
									millie = str(company_df.trip_start[i]).split('.')[1][:3]
								else:
									millie = str(company_df.trip_start[i]).split('.')[1]
								trip_date = ymt_hms+'.'+millie
								#print('2',trip_date)

						#print(ymt_hms+'.'+millie,str(company_df.trip_start[i])[:-3])
						#trip_date = ymt_hms+'.'+millie
						date_lst.append(trip_date)
						#print(trip_date)

					valid_gps = round((1-(missing_gps/total_data_row)) * 100, 2)
					if total_trip_sec == 0:
						data_quality = 0
					else:
						data_quality = round((total_data_row/total_trip_sec) * 100,2)
					val = comp,date_lst,data_size,total_trip_sec,total_data_row,data_quality,missing_gps,valid_gps
					vals.append(val)

		df = pd.DataFrame(vals, columns = cols)
		df.replace(False, 0, inplace=True)
		return df


	def poc_trip(company_imei,trip_start,trip_end):
	    engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
	    trip_query = pd.read_sql_query("\
	      SELECT camera_id, trip_start,\
	      MAX(CASE WHEN st.first_row=1 THEN st.event END) as start_event,\
	      MAX(CASE WHEN st.last_row=1 THEN st.event END) as end_event,\
	      MAX(CASE WHEN st.last_row=2 THEN st.distance END) as end_dist,\
	       AVG(scd_focus_skor) AS avg_focus_skor,\
	    AVG(scd_guard_skor) AS avg_guard_skor,\
	    AVG(scd_speed_skor) AS avg_speed_skor,\
	      (CASE WHEN COALESCE(avg_speed_skor, avg_focus_skor, avg_guard_skor) IS NOT NULL THEN\
	    (SUM(COALESCE(scd_focus_skor,0)) + SUM(COALESCE(scd_speed_skor,0))\
	    + SUM(COALESCE(scd_guard_skor,0))) / \
	    (COUNT(scd_focus_skor) + COUNT(scd_guard_skor)+ COUNT(scd_speed_skor))END) AS overall_skor,\
	      MAX(location) AS location,\
	      COUNT(*) as data_size,\
	      count(case when speed = 0.0 then st.speed END) as idling_cnt,\
	      count(case when event = 'collision' then st.event END) as fcw,\
	      count(case when event = 'departure' then st.event END) as ldw,\
	      count(case when situation = 'accel' then st.situation END) as accel,\
	      count(case when situation = 'brake' then st.situation END) as brake,\
	      count(case when situation = 'stop' then st.situation END) as stop,\
	      count(case when situation = 'front' then st.situation END) as front,\
	      count(case when event = 'sudden' then st.event END) as sudden,\
	      MAX(CASE WHEN st.first_row=1 THEN st.version END) as version\
	      FROM (SELECT camera_id, trip_start, time_stamp,distance, event, \
	      location, speed,version, situation,\
	          DECODE (distance < 0.01, 0, focus_skor) AS scd_focus_skor,\
	          DECODE (distance < 0.01, 0, guard_skor) AS scd_guard_skor,\
	          DECODE (distance < 0.01, 0, speed_skor) AS scd_speed_skor,\
	        ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp ASC) AS first_row, \
	        ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp DESC) AS last_row \
	    FROM carvi_normal_data\
	      WHERE camera_id in "+str(tuple(company_imei))+"\
	      AND trip_start >= '"+trip_start+"' \
	      AND trip_start < '"+trip_end+"') st\
	      GROUP BY camera_id, trip_start\
	      ORDER BY trip_start;", engine)  
	    #trip_query = trip_query.sort_values(['trip_start'])
	    trip_query = trip_query.reset_index()
	    trip_query = trip_query.drop('index',axis = 1)
	    #trip_query = trip_query[trip_query.distance>0.001]
	    return trip_query

	def trip_skor(company_imei,trip_start):
	    engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
	    trip_query = pd.read_sql_query("\
	      SELECT camera_id, trip_start,\
	      COUNT(*) as data_size,\
	      MAX(CASE WHEN st.last_row=2 THEN st.distance END) as total_dist,\
	       AVG(scd_focus_skor) AS avg_focus_skor,\
	    AVG(scd_guard_skor) AS avg_guard_skor,\
	    AVG(scd_speed_skor) AS avg_speed_skor,\
	      (CASE WHEN COALESCE(avg_speed_skor, avg_focus_skor, avg_guard_skor) IS NOT NULL THEN\
	    (SUM(COALESCE(scd_focus_skor,0)) + SUM(COALESCE(scd_speed_skor,0))\
	    + SUM(COALESCE(scd_guard_skor,0))) / \
	    (COUNT(scd_focus_skor) + COUNT(scd_guard_skor)+ COUNT(scd_speed_skor))END) AS overall_skor,\
	      count(case when speed = 0.0 then st.speed END) as idling_cnt,\
	      count(case when event = 'collision' then st.event END) as fcw,\
	      count(case when event = 'departure' then st.event END) as ldw,\
	      count(case when situation = 'accel' then st.situation END) as accel,\
	      count(case when situation = 'brake' then st.situation END) as brake,\
	      count(case when situation = 'stop' then st.situation END) as stop,\
	      count(case when situation = 'front' then st.situation END) as front,\
	      count(case when event = 'sudden' then st.event END) as suddens\
	      FROM (SELECT camera_id, trip_start, time_stamp,distance, event, \
	      location, speed,version, situation,\
	          DECODE (distance < 0.01, 0, focus_skor) AS scd_focus_skor,\
	          DECODE (distance < 0.01, 0, guard_skor) AS scd_guard_skor,\
	          DECODE (distance < 0.01, 0, speed_skor) AS scd_speed_skor,\
	        ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp ASC) AS first_row, \
	        ROW_NUMBER() OVER (PARTITION BY trip_start ORDER BY time_stamp DESC) AS last_row \
	    FROM carvi_normal_data\
	      WHERE camera_id = '"+company_imei+"'\
	      AND trip_start = '"+trip_start+"') st\
	      GROUP BY camera_id, trip_start\
	      ORDER BY trip_start;", engine)  
	    #trip_query = trip_query.sort_values(['trip_start'])
	    trip_query = trip_query.reset_index()
	    trip_query = trip_query.drop('index',axis = 1)
	    #trip_query = trip_query[trip_query.distance>0.001]
	    return trip_query

	def company_api(company_imei,trip_start,trip_end):
	    engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
	    trip_query = pd.read_sql_query("\
	                       SELECT COUNT(*) AS data_cnt,\
	        SUM(kilometers) AS kilometers,\
	        SUM(collision_kilometers) as collision_kilometers,\
	        SUM(collision_kilometers) / SUM(CAST(kilometers AS FLOAT)) AS collision_risk_context,\
	        SUM(events) AS events,\
	        SUM(CAST(collision AS FLOAT)) AS collisions,\
	        SUM(CAST(departure AS FLOAT)) AS departures,\
	        SUM(departure) / CAST(SUM(events) AS FLOAT) AS departure_risk_context,\
	        AVG(CAST(relative_speed AS FLOAT)) AS avg__relative_kph,\
	        MIN(relative_speed) AS min, MAX(relative_speed) AS max,\
	            AVG(relative_speed) AS avg,\
	        SUM(idle_seconds) AS sum__idle_seconds,\
	        SUM(idle_seconds_due_to_front) AS sum__idle_seconds_due_to_front,\
	        SUM(CAST(sudden AS FLOAT)) AS suddens,\
	        sum(brake) as brake,\
	        sum(stop) as stop,\
	        sum(accel) as accel,\
	        sum(change) as change,\
	        sum(rea_ignore) as ignore,\
	        sum(reduction) as reduction,\
	        SUM(brake) / SUM(CAST(sudden AS FLOAT)) AS brake_context,\
	        SUM(stop) / SUM(CAST(sudden AS FLOAT)) AS stop_context,\
	        SUM(accel) / SUM(CAST(sudden AS FLOAT)) AS accel_context,\
	        AVG(left_bias) AS left_bias,\
	        AVG(right_bias) AS right_bias,\
	        AVG(scd_focus_skor) AS avg_focus_skor,\
	        AVG(scd_guard_skor) AS avg_guard_skor,\
	        AVG(scd_speed_skor) AS avg_speed_skor,\
	        (CASE WHEN COALESCE(avg_speed_skor, avg_focus_skor, avg_guard_skor) IS NOT NULL THEN\
	        (SUM(COALESCE(scd_focus_skor,0)) + SUM(COALESCE(scd_speed_skor,0))\
	        + SUM(COALESCE(scd_guard_skor,0))) / \
	        (COUNT(scd_focus_skor) + COUNT(scd_guard_skor)+ COUNT(scd_speed_skor))END) AS overall_skor\
	         FROM  (SELECT *,\
	              NVL(trip_start, (FIRST_VALUE(trip_start\
	                                           IGNORE NULLS) OVER(rows between 10 preceding and 10 following))) AS trip,\
	              NVL((distance - LAG(distance,1) OVER (PARTITION BY camera_id, trip_start\
	                                                    ORDER BY distance, time_stamp)),distance) AS kilometers,\
	              NVL((collision_distance - LAG(collision_distance,1) OVER (PARTITION BY camera_id, trip_start\
	                                                                        ORDER BY distance, time_stamp)),collision_distance) AS collision_kilometers,\
	              NVL(front_speed - speed,NULL) AS relative_speed,\
	              (CASE WHEN (speed  = 0 and hdop > 0 ) THEN 1 END) AS idle_seconds,\
	              (CASE WHEN  (front_distance <= 100 AND speed = 0) THEN 1 END) AS idle_seconds_due_to_front,\
	              DECODE (distance < 0.01, 0, focus_skor) AS scd_focus_skor,\
	              DECODE (distance < 0.01, 0, guard_skor) AS scd_guard_skor,\
	              DECODE (distance < 0.01, 0, speed_skor) AS scd_speed_skor,\
	              DECODE (situation,'brake', 1) AS brake,\
	              DECODE (situation,'stop', 1) AS stop,\
	              DECODE (situation,'accel', 1) AS accel,\
	              DECODE (event,'sudden', 1) AS sudden,\
	              DECODE (event,'departure', 1) AS departure,\
	              DECODE (event,'collision', 1) AS collision,\
	              DECODE (reaction,'change', 1) AS change,\
	              DECODE (reaction,'ignore', 1) AS rea_ignore,\
	              DECODE (reaction,'reduction', 1) AS reduction,\
	              (CASE WHEN (event LIKE 'normal') THEN 0 ELSE 1 END) AS events,\
	              (NTILE(4) OVER(ORDER BY bias DESC)) as bias_quartile,\
	              (percentile_disc(0.25) within group (order by bias) over()) AS left_bias,\
	              (percentile_disc(0.75) within group (order by bias) over()) AS right_bias\
	              FROM carvi_normal_data\
	              WHERE camera_id IN "+str(tuple(company_imei))+"\
	              AND trip_start >= '"+trip_start+"' \
	              AND trip_start < '"+trip_end+"' \
	              ORDER BY time_stamp DESC);", engine)  
	    #trip_query = trip_query.sort_values(['trip_start'])
	    trip_query = trip_query.reset_index()
	    trip_query = trip_query.drop('index',axis = 1)
	    #trip_query = trip_query[trip_query.distance>0.001]
	    return trip_query

	def single_api(company_imei,trip_start,trip_end):
	    engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
	    trip_query = pd.read_sql_query("\
	                      SELECT COUNT(*) AS data_cnt,\
	        SUM(kilometers) AS kilometers,\
	        SUM(collision_kilometers) as collision_kilometers,\
	        SUM(collision_kilometers) / SUM(CAST(kilometers AS FLOAT)) AS collision_risk_context,\
	        SUM(events) AS events,\
	        SUM(CAST(collision AS FLOAT)) AS collisions,\
	        SUM(CAST(departure AS FLOAT)) AS departures,\
	        SUM(departure) / CAST(SUM(events) AS FLOAT) AS departure_risk_context,\
	        AVG(CAST(relative_speed AS FLOAT)) AS avg__relative_kph,\
	        MIN(relative_speed) AS min, MAX(relative_speed) AS max,\
	            AVG(relative_speed) AS avg,\
	        SUM(idle_seconds) AS sum__idle_seconds,\
	        SUM(idle_seconds_due_to_front) AS sum__idle_seconds_due_to_front,\
	        SUM(CAST(sudden AS FLOAT)) AS suddens,\
	        sum(brake) as brake,\
	        sum(stop) as stop,\
	        sum(accel) as accel,\
	        sum(change) as change,\
	        sum(rea_ignore) as ignore,\
	        sum(reduction) as reduction,\
	        SUM(brake) / SUM(CAST(sudden AS FLOAT)) AS brake_context,\
	        SUM(stop) / SUM(CAST(sudden AS FLOAT)) AS stop_context,\
	        SUM(accel) / SUM(CAST(sudden AS FLOAT)) AS accel_context,\
	        AVG(left_bias) AS left_bias,\
	        AVG(right_bias) AS right_bias,\
	        AVG(scd_focus_skor) AS avg_focus_skor,\
	        AVG(scd_guard_skor) AS avg_guard_skor,\
	        AVG(scd_speed_skor) AS avg_speed_skor,\
	        (CASE WHEN COALESCE(avg_speed_skor, avg_focus_skor, avg_guard_skor) IS NOT NULL THEN\
	        (SUM(COALESCE(scd_focus_skor,0)) + SUM(COALESCE(scd_speed_skor,0))\
	        + SUM(COALESCE(scd_guard_skor,0))) / \
	        (COUNT(scd_focus_skor) + COUNT(scd_guard_skor)+ COUNT(scd_speed_skor))END) AS overall_skor\
	         FROM  (SELECT *,\
	              NVL(trip_start, (FIRST_VALUE(trip_start\
	                                           IGNORE NULLS) OVER(rows between 10 preceding and 10 following))) AS trip,\
	              NVL((distance - LAG(distance,1) OVER (PARTITION BY camera_id, trip_start\
	                                                    ORDER BY distance, time_stamp)),distance) AS kilometers,\
	              NVL((collision_distance - LAG(collision_distance,1) OVER (PARTITION BY camera_id, trip_start\
	                                                                        ORDER BY distance, time_stamp)),collision_distance) AS collision_kilometers,\
	              NVL(front_speed - speed,NULL) AS relative_speed,\
	              (CASE WHEN (speed  = 0 and hdop > 0 ) THEN 1 END) AS idle_seconds,\
	              (CASE WHEN  (front_distance <= 100 AND speed = 0) THEN 1 END) AS idle_seconds_due_to_front,\
	              DECODE (distance < 0.01, 0, focus_skor) AS scd_focus_skor,\
	              DECODE (distance < 0.01, 0, guard_skor) AS scd_guard_skor,\
	              DECODE (distance < 0.01, 0, speed_skor) AS scd_speed_skor,\
	              DECODE (situation,'brake', 1) AS brake,\
	              DECODE (situation,'stop', 1) AS stop,\
	              DECODE (situation,'accel', 1) AS accel,\
	              DECODE (event,'sudden', 1) AS sudden,\
	              DECODE (event,'departure', 1) AS departure,\
	              DECODE (event,'collision', 1) AS collision,\
	              DECODE (reaction,'change', 1) AS change,\
	              DECODE (reaction,'ignore', 1) AS rea_ignore,\
	              DECODE (reaction,'reduction', 1) AS reduction,\
	              (CASE WHEN (event LIKE 'normal') THEN 0 ELSE 1 END) AS events,\
	              (NTILE(4) OVER(ORDER BY bias DESC)) as bias_quartile,\
	              (percentile_disc(0.25) within group (order by bias) over()) AS left_bias,\
	              (percentile_disc(0.75) within group (order by bias) over()) AS right_bias\
	              FROM carvi_normal_data\
	              WHERE camera_id = '"+company_imei+"'\
	              AND trip_start >= '"+trip_start+"' \
	              AND trip_start < '"+trip_end+"' \
	              ORDER BY time_stamp DESC);", engine)  
	    #trip_query = trip_query.sort_values(['trip_start'])
	    trip_query = trip_query.reset_index()
	    trip_query = trip_query.drop('index',axis = 1)
	    #trip_query = trip_query[trip_query.distance>0.001]
	    return trip_query


	def company_api_v(company_imei,trip_start,trip_end):
	    engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
	    trip_query = pd.read_sql_query("\
	        SELECT  COUNT(*) AS data_cnt,\
	                SUM(kilometers) AS kilometers,\
	                AVG(scd_guard_skor) AS avg_guard_skor,\
	                AVG(scd_focus_skor) AS avg_focus_skor,\
	                AVG(scd_speed_skor) AS avg_speed_skor,\
	                (CASE WHEN COALESCE(avg_speed_skor, avg_focus_skor, avg_guard_skor) IS NOT NULL THEN\
	                (SUM(COALESCE(scd_focus_skor,0)) + SUM(COALESCE(scd_speed_skor,0))\
	                    + SUM(COALESCE(scd_guard_skor,0))) / \
	                    (COUNT(scd_focus_skor) + COUNT(scd_guard_skor)+ COUNT(scd_speed_skor))END) AS overall_skor,\
	                AVG(relative_speed) AS avg,\
	                MIN(relative_speed) AS min,\
	                MAX(relative_speed) AS max,\
	                SUM(idle_seconds) AS sum__idle_seconds,\
	                SUM(idle_seconds_due_to_front) AS sum__idle_seconds_due_to_front,\
	                SUM(events) AS events,\
	                AVG(left_bias) AS left_bias,\
	                AVG(right_bias) AS right_bias,\
	                SUM(CAST(collision AS FLOAT)) AS collisions,\
	                SUM(CAST(departure AS FLOAT)) AS departures,\
	                SUM(CAST(sudden AS FLOAT)) AS suddens,\
	                sum(accel) as accel,\
	                sum(brake) as brake,\
	                sum(stop) as stop,\
	                sum(departure_left) as departure_left,\
	                sum(departure_right) as departure_right,\
	                count(reaction) as reaction,\
	                sum(rea_ignore) as ignore,\
	                sum(rea_recover) as recover,\
	                sum(reduction) as reduction,\
	                SUM(collision_kilometers) as collision_kilometers\
	         FROM  (SELECT *,\
	              NVL(trip_start, (FIRST_VALUE(trip_start\
	                                           IGNORE NULLS) OVER(rows between 10 preceding and 10 following))) AS trip,\
	              NVL((distance - LAG(distance,1) OVER (PARTITION BY camera_id, trip_start\
	                                                    ORDER BY distance, time_stamp)),distance) AS kilometers,\
	              NVL((collision_distance - LAG(collision_distance,1) OVER (PARTITION BY camera_id, trip_start\
	                                                                        ORDER BY distance, time_stamp)),collision_distance) AS collision_kilometers,\
	              NVL(front_speed - speed,NULL) AS relative_speed,\
	              (CASE WHEN (speed  = 0 and hdop > 0 ) THEN 1 END) AS idle_seconds,\
	              (CASE WHEN  (front_distance <= 100 AND speed = 0) THEN 1 END) AS idle_seconds_due_to_front,\
	              DECODE (distance < 0.01, 0, focus_skor) AS scd_focus_skor,\
	              DECODE (distance < 0.01, 0, guard_skor) AS scd_guard_skor,\
	              DECODE (distance < 0.01, 0, speed_skor) AS scd_speed_skor,\
	              DECODE (situation,'brake', 1) AS brake,\
	              DECODE (situation,'stop', 1) AS stop,\
	              DECODE (situation,'accel', 1) AS accel,\
	              DECODE (event,'sudden', 1) AS sudden,\
	              DECODE (event,'departure', 1) AS departure,\
	              DECODE (event,'collision', 1) AS collision,\
	              DECODE (direction,'left', 1) AS departure_left,\
	              DECODE (direction,'right', 1) AS departure_right,\
	              DECODE (reaction,'change', 1) AS change,\
	              DECODE (reaction,'ignore', 1) AS rea_ignore,\
	              DECODE (reaction,'recover', 1) AS rea_recover,\
	              DECODE (reaction,'reduction', 1) AS reduction,\
	              (CASE WHEN (event LIKE 'normal') THEN 0 ELSE 1 END) AS events,\
	              (NTILE(4) OVER(ORDER BY bias DESC)) as bias_quartile,\
	              (percentile_disc(0.25) within group (order by bias) over()) AS left_bias,\
	              (percentile_disc(0.75) within group (order by bias) over()) AS right_bias\
	              FROM carvi_normal_data\
	              WHERE camera_id IN "+str(tuple(company_imei))+"\
	              AND trip_start >= '"+trip_start+"' \
	              AND trip_start < '"+trip_end+"' \
	              ORDER BY time_stamp DESC);", engine)  
	    #trip_query = trip_query.sort_values(['trip_start'])
	    trip_query = trip_query.reset_index()
	    trip_query = trip_query.drop('index',axis = 1)
	    #trip_query = trip_query[trip_query.distance>0.001]
	    return trip_query

	def single_api_v(company_imei,trip_start,trip_end):
	    engine = sqlalchemy.create_engine('postgresql+psycopg2://carvi:Carvigogo1@dev-carvi-redshift-from-firehose.cgv3wupnu6mw.us-west-2.redshift.amazonaws.com:5439/dev')
	    trip_query = pd.read_sql_query("\
	                      SELECT  COUNT(*) AS data_cnt,\
	                SUM(kilometers) AS kilometers,\
	                AVG(scd_guard_skor) AS avg_guard_skor,\
	                AVG(scd_focus_skor) AS avg_focus_skor,\
	                AVG(scd_speed_skor) AS avg_speed_skor,\
	                (CASE WHEN COALESCE(avg_speed_skor, avg_focus_skor, avg_guard_skor) IS NOT NULL THEN\
	                (SUM(COALESCE(scd_focus_skor,0)) + SUM(COALESCE(scd_speed_skor,0))\
	                    + SUM(COALESCE(scd_guard_skor,0))) / \
	                    (COUNT(scd_focus_skor) + COUNT(scd_guard_skor)+ COUNT(scd_speed_skor))END) AS overall_skor,\
	                AVG(relative_speed) AS avg,\
	                MIN(relative_speed) AS min,\
	                MAX(relative_speed) AS max,\
	                SUM(idle_seconds) AS sum__idle_seconds,\
	                SUM(idle_seconds_due_to_front) AS sum__idle_seconds_due_to_front,\
	                SUM(events) AS events,\
	                AVG(left_bias) AS left_bias,\
	                AVG(right_bias) AS right_bias,\
	                SUM(CAST(collision AS FLOAT)) AS collisions,\
	                SUM(CAST(departure AS FLOAT)) AS departures,\
	                SUM(CAST(sudden AS FLOAT)) AS suddens,\
	                sum(accel) as accel,\
	                sum(brake) as brake,\
	                sum(stop) as stop,\
	                sum(departure_left) as departure_left,\
	                sum(departure_right) as departure_right,\
	                count(reaction) as reaction,\
	                sum(rea_ignore) as ignore,\
	                sum(rea_recover) as recover,\
	                sum(reduction) as reduction,\
	                SUM(collision_kilometers) as collision_kilometers\
	         FROM  (SELECT *,\
	              NVL(trip_start, (FIRST_VALUE(trip_start\
	                                           IGNORE NULLS) OVER(rows between 10 preceding and 10 following))) AS trip,\
	              NVL((distance - LAG(distance,1) OVER (PARTITION BY camera_id, trip_start\
	                                                    ORDER BY distance, time_stamp)),distance) AS kilometers,\
	              NVL((collision_distance - LAG(collision_distance,1) OVER (PARTITION BY camera_id, trip_start\
	                                                                        ORDER BY distance, time_stamp)),collision_distance) AS collision_kilometers,\
	              NVL(front_speed - speed,NULL) AS relative_speed,\
	              (CASE WHEN (speed  = 0 and hdop > 0 ) THEN 1 END) AS idle_seconds,\
	              (CASE WHEN  (front_distance <= 100 AND speed = 0) THEN 1 END) AS idle_seconds_due_to_front,\
	              DECODE (distance < 0.01, 0, focus_skor) AS scd_focus_skor,\
	              DECODE (distance < 0.01, 0, guard_skor) AS scd_guard_skor,\
	              DECODE (distance < 0.01, 0, speed_skor) AS scd_speed_skor,\
	              DECODE (situation,'brake', 1) AS brake,\
	              DECODE (situation,'stop', 1) AS stop,\
	              DECODE (situation,'accel', 1) AS accel,\
	              DECODE (event,'sudden', 1) AS sudden,\
	              DECODE (event,'departure', 1) AS departure,\
	              DECODE (event,'collision', 1) AS collision,\
	              DECODE (direction,'left', 1) AS departure_left,\
	              DECODE (direction,'right', 1) AS departure_right,\
	              DECODE (reaction,'change', 1) AS change,\
	              DECODE (reaction,'ignore', 1) AS rea_ignore,\
	              DECODE (reaction,'recover', 1) AS rea_recover,\
	              DECODE (reaction,'reduction', 1) AS reduction,\
	              (CASE WHEN (event LIKE 'normal') THEN 0 ELSE 1 END) AS events,\
	              (NTILE(4) OVER(ORDER BY bias DESC)) as bias_quartile,\
	              (percentile_disc(0.25) within group (order by bias) over()) AS left_bias,\
	              (percentile_disc(0.75) within group (order by bias) over()) AS right_bias\
	              FROM carvi_normal_data\
	              WHERE camera_id = '"+company_imei+"'\
	              AND trip_start >= '"+trip_start+"' \
	              AND trip_start < '"+trip_end+"' \
	              ORDER BY time_stamp DESC);", engine)  
	    #trip_query = trip_query.sort_values(['trip_start'])
	    trip_query = trip_query.reset_index()
	    trip_query = trip_query.drop('index',axis = 1)
	    #trip_query = trip_query[trip_query.distance>0.001]
	    return trip_query

	def trip_information(df_redshift):

	    #trip_start = company_trip_list['trip_start'].iloc[index_trip]
	    #df_redshift = jay.redshift_data(imei, trip_start)
	    trip_start = df_redshift.trip_start.iloc[0]
	    imei = df_redshift.camera_id.iloc[0]
	    tz_ind = df_redshift.index[df_redshift.lat>0][0]
	    tz = jay.local_timezone(tz_ind, df_redshift)
	    firmware = df_redshift['version'].iloc[0].split(',')[0][2:-1]
	    print(colored("Firmware Version", "blue", attrs = ['bold']), colored(firmware, "red"))
	    print()

	    event_start = df_redshift.head()['event'].iloc[0]
	    event_end = df_redshift.tail()['event'].iloc[-1]
	    trip_end = df_redshift.time_stamp.tail().iloc[-1]
	    start = 'Trip Start'
	    end = 'Trip End'

	    utc_trip_start = jay.convert_utc_trip_start(trip_start,tz)
	    utc_start_year, utc_start_hour, utc_trip_start = jay.local_utc_time(event_start, trip_start, start,utc_trip_start,tz)
	    print()
	    utc_trip_end = jay.convert_utc_trip_start(trip_end,tz)
	    utc_end_year, utc_end_hour, utc_trip_end = jay.local_utc_time(event_end, trip_end, end,utc_trip_end,tz)

	    jay.trip_info(df_redshift)#, imei)
	    
	    return jay.google_map(df_redshift)
	

	def soso(company_trip_list,index):

	    index_timezone = index
	    imei = company_trip_list.camera_id[index]
	    tz = jay.local_timezone(index_timezone, company_trip_list)
	    index_trip = index

	    trip_start = company_trip_list['trip_start'].iloc[index_trip]
	    df_redshift = jay.redshift_data(imei, trip_start)
	    firmware = df_redshift['version'].iloc[0].split(',')[0][2:-1]
	    print(colored("Firmware Version", "blue", attrs = ['bold']), colored(firmware, "red"))
	    print()

	    event_start = df_redshift.head()['event'].iloc[0]
	    event_end = df_redshift.tail()['event'].iloc[-1]
	    trip_end = df_redshift.time_stamp.tail().iloc[-1]
	    start = 'Trip Start'
	    end = 'Trip End'

	    utc_trip_start = jay.convert_utc_trip_start(trip_start,tz)
	    utc_start_year, utc_start_hour, utc_trip_start = jay.local_utc_time(event_start, trip_start, start,utc_trip_start,tz)
	    print()
	    utc_trip_end = jay.convert_utc_trip_start(trip_end,tz)
	    utc_end_year, utc_end_hour, utc_trip_end = jay.local_utc_time(event_end, trip_end, end,utc_trip_end,tz)

	    jay.trip_info(df_redshift)
	    
	    return jay.google_map(df_redshift)
