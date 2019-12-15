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
		gmaps.configure(api_key='your_google_api_key')
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
