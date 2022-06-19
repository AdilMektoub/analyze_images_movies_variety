# import the necessary packages
from collections import Counter

import numpy as np
import cv2
from networkx.drawing.tests.test_pylab import plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from webcolors import rgb_to_name
import webcolors
from scipy.spatial import KDTree
from webcolors import css3_hex_to_names,hex_to_rgb

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)

	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()

	# return the histogram
	return hist

def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0

	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		#print("percent ",percent)
		#print("color ",color)
		rgb_tuple = color
		print("rgb_tuple", rgb_tuple)
		#print(int(r,g,b))

		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	# return the bar chart
	#print(rgb_to_name((20,20,20), spec='css3'))
	return bar,rgb_tuple


def convert_rgb_to_names(rgb_tuple):
	# a dictionary of all the hex and their respective names in css3
	css3_db = css3_hex_to_names
	names = []
	rgb_values = []
	for color_hex, color_name in css3_db.items():
		names.append(color_name)
		rgb_values.append(hex_to_rgb(color_hex))

	kdt_db = KDTree(rgb_values)
	distance, index = kdt_db.query(rgb_tuple)
	return f'closest match: {names[index]}'


# # def get_colour_name(rgb_triplet):
# # 	named_color = rgb_to_name((255, 0, 0), spec='css3')
#
# # def get_colour_name(rgb_triplet):
# #     min_colours = {}
# #     for key, name in webcolors.css21_hex_to_names.items():
# #         r_c, g_c, b_c = webcolors.hex_to_rgb(key)
# #         rd = (r_c - rgb_triplet[0]) ** 2
# #         gd = (g_c - rgb_triplet[1]) ** 2
# #         bd = (b_c - rgb_triplet[2]) ** 2
# #         min_colours[(rd + gd + bd)] = name
# #     return min_colours[min(min_colours.keys())]
# def rgb2hex(c):
#     return "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2]))  # format(int(c[0]), int(c[1]), int(c[2]))
#
# def hex2name(c):
#     h_color = '#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2]))
#     try:
#         nm = webcolors.hex_to_name(h_color, spec='css3')
#     except ValueError as v_error:
#         print("{}".format(v_error))
#         rms_lst = []
#         for img_clr, img_hex in webcolors.CSS3_NAMES_TO_HEX.items():
#             cur_clr = webcolors.hex_to_rgb(img_hex)
#             rmse = np.sqrt(mean_squared_error(c, cur_clr))
#             rms_lst.append(rmse)
#
#         closest_color = rms_lst.index(min(rms_lst))
#
#         nm = list(webcolors.CSS3_NAMES_TO_HEX.items())[closest_color][0]
#     return nm
# img = cv2.imread("tt0084058.jpg")
# img2 = img.reshape(img.shape[0] * img.shape[1], 3)
# color = KMeans(n_clusters=3)
# lbl = color.fit_predict(img2)
# cnt = Counter(lbl)
# center_color = color.cluster_centers_
# ord_color = [center_color[i] for i in cnt.keys()]
# hex_color = [rgb2hex(ord_color[i]) for i in cnt.keys()]
# lbl_color = [hex2name(ord_color[i]) for i in cnt.keys()]
# plt.pie(cnt.values(), labels=lbl_color, colors=hex_color)
# plt.show()
