from scipy.io import loadmat
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def make_heatmap(im, dt, cbar = False):
	sns.heatmap(im, cmap = 'coolwarm', cbar = cbar)
	if dt is None:
		new_title = 'No date'
	else:
		dt = stamp_to_date(int(dt[0]))
		new_title = str(dt[1])+'-'+str(dt[0])+'-2018 '+ str(dt[2])+ ':'+ str(dt[3])
	plt.title(new_title)
	plt.yticks(np.arange(28, step = 4),np.arange(3, 30, step = 4))
	xinterval= np.timedelta64(30,'s')
	plt.xticks(np.arange(im.shape[1], step = 10), [(xinterval*i) / np.timedelta64(1, 'm') for i in np.arange(im.shape[1], step = 10)])
	plt.xlabel("Time (minutes)")
	plt.ylabel("Depth")
	plt.ioff()
	plt.show()
			
def make_12_heatmaps(im, title = 'Wake Samples', cbar = False, save = False, filename = None):
	fig, axs = plt.subplots(4, 3, figsize=(25,15))
	fig.suptitle(title, fontweight="bold", size=20)
	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	for i in range(4):
		for j in range(3):
			ax = axs[i,j]
			sns.heatmap(im[i*3+j], cmap = 'coolwarm', ax = ax, cbar = cbar)            
	xinterval = np.timedelta64(30,'s')
	plt.setp(axs, xticks=np.arange(im.shape[2], step = 10), 
			 xticklabels=[(xinterval*j) / np.timedelta64(1, 'm') for j in np.arange(im.shape[2], step = 10)],
			 yticks=np.arange(28, step = 4), yticklabels = np.arange(3, 30, step = 4))
	fig.text(0.45, 0.01, 'Time (minutes)', ha='center',  size=20)
	fig.text(-0.01, 0.5, 'Depth (meters)', va='center', rotation='vertical',  size=20)
	if save:
		plt.savefig(filename)
	plt.show()
	

def stamp_to_date(st):
	month = 8
	day = st // (3600*24) + 28
	hour = st % (3600*24) // 3600 + 11
	mins = st % (3600*24) % 3600 / 60
	if hour >= 24:
		hour -= 24
		day += 1
	if day > 31:
		day -= 31
		month +=1
	return month, day, hour, mins
	

def date_to_stamp(dt):
	month, day, hour, minute = dt
	stamp = 0
	if month == 9:
		stamp += 24*3600*4
		stamp -= 11*3600
		stamp += (day-1)*24*3600
		stamp += (hour)*3600
		stamp += (minute)*60
	if month == 8:
		stamp += (day-28)*24*3600
		stamp += (hour-11)*3600
		stamp += (minute)*60
	return stamp
	
def frames_by_timestamps(data_series, wake_start_ts):
	wake_samples = []
	for ws in wake_start_ts:
		ind = np.where(data_series[:,-1,:]==ws)[1]
		if ind.size !=0:
			ind2 = int(ind[0])+60
			wake_samples.append(data_series[:,:,int(ind[0]):ind2])
	wake_samples = np.array(wake_samples)
	return wake_samples
	
def get_files():
	m = ['08_']*8 + ['09_']*49
	d_n = ['day_', 'night_']*28 + ['day_']
	path = 'Data\\Matlab files'
	day_list = []
	for i in range(28, 31+26):
		if i > 31:
			dd =  i % 32 + 1
			if dd < 10:
				dd = '0' + str(dd)
			else: dd = str(dd)
		else:
			dd = str(i % 32)
		day_list.append(dd)
	d = []
	yesterday = '__'
	for today in day_list:
		d.append(yesterday+'_'+today)
		d.append(today)
		yesterday = today
	d = d[1:]
	file_names = []
	for i in range(57):
		file_name = d_n[i]+m[i]+d[i]+'_pythonfile.mat'
		file_names.append(file_name)
	full_data_5 = loadmat(path + "\\" + file_names[0])["a5m"]
	full_data_4 = loadmat(path + "\\" + file_names[0])["a4m"]
	full_data_3 = loadmat(path + "\\" + file_names[0])["a3m"]
	full_data_1 = loadmat(path + "\\" + file_names[0])["a1m"]
	full_times = loadmat(path + "\\" + file_names[0])['tm']
	start_time = full_times[0,0]
	full_times = np.round((full_times - start_time)*24*3600)
	day_night = np.ones(full_times.shape) # 1 - day, 0 - night
	day_count = full_times.shape[1]
	night_count = 0
	for i, file in enumerate(file_names[1:]):
		mat_file = loadmat(path + "\\" + file)
		full_data_5 = np.append(full_data_5, mat_file["a5m"], axis = 0)
		full_data_4 = np.append(full_data_4, mat_file["a4m"], axis = 0)
		full_data_3 = np.append(full_data_3, mat_file["a3m"], axis = 0)
		full_data_1 = np.append(full_data_1, mat_file["a1m"], axis = 0)
		# Convert timestamps from days to secs
		# Assuming first stamp to be from 11.00, 28 August 2018
		times = np.round((mat_file["tm"] - start_time)*24*3600)
		full_times = np.append(full_times, times)
		if i % 2 == 0:
			night_count += times.shape[1]
			dn = np.zeros(times.shape)
		else:
			day_count += times.shape[1]
			dn = np.ones(times.shape)
		day_night = np.append(day_night, dn)
	max_response = np.array([full_data_1, full_data_3, full_data_4, full_data_5]).max()
	data_series_5 = np.append(np.flip(full_data_5, 1)/max_response, full_times.reshape(full_times.shape[0], 1), axis = 1)
	data_series_4 = np.append(np.flip(full_data_4, 1)/max_response, full_times.reshape(full_times.shape[0], 1), axis = 1)
	data_series_3 = np.append(np.flip(full_data_3, 1)/max_response, full_times.reshape(full_times.shape[0], 1), axis = 1)
	data_series_1 = np.append(np.flip(full_data_1, 1)/max_response, full_times.reshape(full_times.shape[0], 1), axis = 1)
	data_series_5 = np.append(data_series_5, day_night.reshape(day_night.shape[0], 1), axis = 1).transpose()
	data_series_4 = np.append(data_series_4, day_night.reshape(day_night.shape[0], 1), axis = 1).transpose()
	data_series_3 = np.append(data_series_3, day_night.reshape(day_night.shape[0], 1), axis = 1).transpose()
	data_series_1 = np.append(data_series_1, day_night.reshape(day_night.shape[0], 1), axis = 1).transpose()
	data_series = np.array([data_series_1, data_series_3, data_series_4, data_series_5])
	#print('day data: ', day_count)
	#print('night data: ', night_count)
	return data_series
	
def get_wakes():
	#Reading identified wake time stamps
	all_wakes = pd.read_csv("All_wakes.csv", encoding = "utf-8")
	all_wakes['wake_start'] = 0
	all_wakes.loc[all_wakes['month'] == 8, 'wake_start'] = (all_wakes['day']-28)*24*3600 + (all_wakes['hour']-11)*3600 + all_wakes['min']*60
	all_wakes.loc[all_wakes['month'] == 9, 'wake_start'] = (all_wakes['day']+2)*24*3600 + (all_wakes['hour']+13)*3600 + all_wakes['min']*60
	wake_start = np.array(all_wakes['wake_start'])
	return wake_start