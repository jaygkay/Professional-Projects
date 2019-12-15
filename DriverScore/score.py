def speed_skor(speed, front_speed, speed_limit = 120):
    ktm = 1.60934
    std_dev = ktm *10.0

    # if there is no front car, front_speed = None
    max_skor = 100 # Adjustable maximum skor
    # assumes driver is going speed limit or near front_car's speed
    mu_relative = 0.0 

    if front_speed != None:
        if front_speed > 120: # if front_speed is too high
            front_speed = speed_limit # adjust front_speed to speed limit

        if speed > speed_limit: # if speed > speed_limit, put heavier penalty
            relative_speed = (front_speed - speed)*2.5 - (speed - speed_limit)
        else:
            relative_speed = (front_speed - speed)*2.5

    else: # if there is no front car
        relative_speed = speed_limit - speed

    # if relative_speed is positive, skor = 100
    if relative_speed >= 0:
        return 100
    else:
        skor = max_skor * e**(-1 * ((relative_speed - mu_relative)**2 / (2.0 * std_dev**2))) 
        return round(skor,2)

def focus_skor(bias, continuity):
	bias = float(bias)
	'''provides focus skor with gussian function of driver's bias and continuity '''
	mu_bias = 0.0 # Adjustable average bias, assumes all drivers typically stay in center of lane
	std_dev = 4.0 # Adjustable standard deviaiton, primary way to adjust curve (higher = higher skor)
	#     elif np.isnan(continuity): # No LDW 
	max_skor = 100

	if continuity != None:
		if continuity in [0,'Lane Change/Recenter']:
			skor = max_skor * e**(-1 * ((bias - mu_bias)**2 / (2.0 * std_dev**2))) * 0.8 # abs(con - 1)
		elif continuity in [1,'Lane Straddling']:
			skor = max_skor * e**(-1 * ((bias - mu_bias)**2 / (2.0 * std_dev**2))) * 0.5
			
	else:
		skor = max_skor * e**(-1 * ((bias - mu_bias)**2 / (2.0 * std_dev**2)))
	return round(skor)


def guard_skor(prewarn, reaction_c, situation_s, front_distance, front_reason, ttc,
               front_reduction_speed, time_stamp, mileage, collision_mileage):
    '''provides guard skor with exponential function of collision_mileage/mileage and multiple features
    such as reaction to warnings, suddens, tailgating, and daytime vs nighttime driving'''
    max_skor = 100.0
    lamb = 4.0 # Adjustable lambda of exponential function, lower the higher possible skor
        # Exponential function of collision ratio
    collision_skor = max_skor * e**(
        -lamb * (float(collision_mileage) / float(mileage))) 

    # Create list of skors from categorical variables
    skor = [] 
    if prewarn != None:
	    if prewarn in [0,False]:
	        if reaction_c in ['001','ignored']: 
	            skor.append(0.0)
	        # elif reaction_c in ['002','Decelerate']:
	        #     skor.append(80.0)
	        else:
	            skor.append(50.0)
	    elif prewarn in [1, True]:
	        if reaction_c in ['001','ignored']: 
	            skor.append(0.0)
	        # elif reaction_c in ['002','Decelerate']:
	        #     skor.append(60.0)
	        else:
	            skor.append(30.0)
          
    if situation_s != None:  
	    if situation_s in ["001","002",'brake', 'stop']: #sudden stop/brake
	        if front_reason in [1, True]: #positive skor for sudden stop/break with due to front_car
	            skor.append(90.0)
	        else: 
	            skor.append(50.0)
	    if (situation_s in ['003','accel']) and (front_reason in [1, True]): # one of the worst scenarios
	        skor.append(0.0)
	    else:
	        skor.append(70.0)

    # based on ttc and front_distance we will penalize guard_skor 
    if ttc != None:
      ttc_alpha = 0.05
      skor.append(max_skor * e**(
            -ttc_alpha * (float(front_distance) / float(ttc))) )
    
    # If front car reduces speed, penalize driver for tailgating
    if front_reduction_speed != None:
	    if front_reduction_speed > 0: 
	        tsmp = dt.datetime.strptime(time_stamp, '%Y-%m-%d %H:%M:%S').time()
	        if (tsmp >= dt.time(hour=0, minute=0)) and (tsmp <= dt.time(hour=4, minute=0)):
	            # Penalize tailgating at night with logistic function of front_distance 
	            # Note distane of 100 or more will be the "similar" curve as null front_reduction_speed
	            skor.append(max_skor / (
	                1 + e**-(0.08*(float(front_distance) - (100.0)))))
	        else:
	            skor.append(max_skor / (1+e**-(0.08*(front_distance - (50.0))))) 
	            # Penalize tailgating during day with logistic function of front_distance
	            # Note distane of 100 (with front_reduction_speed >0) will be same as distance 
	            # of 50 (with front_reduction_speed >0) during the day
	            # Note distance of 150 or more will be the "similar" curve as null front_reduction_speed
	  
    if len(skor) == 0:
      return collision_skor
    else:         
      final_skor = (sum(skor) / len(skor)) / 100.0 * collision_skor    
      return round(final_skor)
