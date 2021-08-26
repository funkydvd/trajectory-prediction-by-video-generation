import json
import xml.etree.ElementTree as ET
import cv2
import math

gt_xml = "./cvat_rezs/gt_annotations.xml"
annotated_xml = ["./cvat_rezs/prednet_annotations.xml", "./cvat_rezs/vp_annotations.xml","./cvat_rezs/seg2vid_annotations.xml"]
predicted_json = ["./yolo_data/result_pred.json", "./yolo_data/result_vp.json","./yolo_data/result_seg.json"]
depth_folders = ["./monodepth2_data/final_datasets_paper/prednet_small/", "./monodepth2_data/final_datasets_paper/videoPrediction_small/", "./monodepth2_data/final_datasets_paper/seg2vid_small/"]
segm_folders = ["./fcn_data/prednet_small_fcn/", "./fcn_data/vp_small_fcn","./fcn_data/seg2vid_small_fcn"]
raws_folders = ["./raw_data/prednet_small/","./raw_data/seg2vid_small","./raw_data/videoPrediction_small"] 

# annotated gt vs annotated pred - rmse and depth (day/dusk/night)
# annotated gt vs annotated yolo -rmse and depth (day/dusk/night)
# annotated gt vs annotated yolo+ segm - rmse and depth (day/dusk/night including car size)
# rmse system vs trackNPred
# speed?

tree = ET.parse(gt_xml)
root = tree.getroot()

tree2 = ET.parse(annotated_xml[0])
root2 = tree2.getroot()


classes = [0, 100, 250, 500, 750, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 100000, 1000000]
nrs  = []
scor = []

daytime_map = {}
daytime_map['day'] = 1
daytime_map['dusk'] = 2
daytime_map['night'] = 3

for i in range(3):
	nrs.append([])
	scor.append([])
	for j in range(14):
		nrs[i].append(0)
		scor[i].append(0)
images = {}
for child in root:
#	print(child.attrib)
	data = ""
	if 'name' in child.attrib:
		data = child.attrib['name'].split("_")
		path = data[0] + "/" + data[1] +"/" + data[2]
		if data[0] in images:
			if data[1] in images[data[0]]:
				images[data[0]][data[1]].append(data[2])
			else:
				images[data[0]][data[1]] = [data[2]]
		else:
			images[data[0]] = {}
			images[data[0]][data[1]] = [data[2]]
		
ord1 = {}
for i1 in images:
	for i2 in images[i1]:
		ind1 = 0
		for x in sorted(images[i1][i2]):
			ind1 = ind1+1
			if i1 in ord1:
				if i2 in ord1[i1]:
					ord1[i1][i2][x] = ind1
				else:
					ord1[i1][i2] = {}
					ord1[i1][i2][x] = ind1
			else:
				ord1[i1] = {}
				ord1[i1][i2] = {}
				ord1[i1][i2][x] = ind1
				
print(ord1)




scor1 = [0,0,0,0]
numbers1 = [0,0,0,0]

scor1_depth = [0,0,0,0]
numbers1_depth = [0,0,0,0]

scor1_depth_predicted = [0,0,0,0]
numbers1_depth_predicted = [0,0,0,0]


scor1_segm =    [[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]
numbers1_segm = [[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]


scor1_segm_fcn =    [0,0,0,0,0,0,0,0,0,0,0]
numbers1_segm_fcn = [0,0,0,0,0,0,0,0,0,0,0]

scor1_segm_depth =    [[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]
numbers1_segm_depth = [[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]



scor1_segm_depth_predicted = [[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]
numbers1_segm_depth_predicted = [[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]


scor1_segm_fcn_depth =    [0,0,0,0,0,0,0,0,0,0,0]
numbers1_segm_fcn_depth = [0,0,0,0,0,0,0,0,0,0,0]


scor1_segm_fcn_depth_predicted =    [0,0,0,0,0,0,0,0,0,0,0]
numbers1_segm_fcn_depth_predicted = [0,0,0,0,0,0,0,0,0,0,0]


scor1_segm_fcn_classes = []
numbers1_segm_classes = []
for ii in range(len(classes)):
	scor1_segm_fcn_classes.append([0,0,0,0,0,0,0,0,0,0,0])
	numbers1_segm_classes.append([0,0,0,0,0,0,0,0,0,0,0])
	
scor1_segm_classes = []
numbers1_segm_fcn_classes = []
for ii in range(len(classes)):
	scor1_segm_classes.append([0,0,0,0,0,0,0,0,0,0,0])
	numbers1_segm_fcn_classes.append( [0,0,0,0,0,0,0,0,0,0,0])

for child in root:
#	print(child.attrib)
	data = ""
	if 'name' in child.attrib:
		data = child.attrib['name'].split("_")
		path = data[0] + "/" + data[1] +"/" + data[2]
		print(path)
		for child_1 in root2:
			if 'name' in child_1.attrib:
				data2 = child_1.attrib['name'].split("_")
				path2 = data2[0] + "/" + data2[1] +"/" + data2[2] #+ "_" +data2[3] + "_" + data2[4] + "_" + data2[5] + "_" + data2[6] 	
				if data2[0] == data[0] and data2[1] == data[1]:
					if int(data2[2][0])-4 == ord1[data[0]][data[1]][data[2]]:
						#print(path)						
						used_obj = []
						for child2 in child:
							used_obj.append(0)
						
						numeimg = './raw_data/prednet_small/' + path2
						img_default = cv2.imread(numeimg, 0)
								
						depth_file1 = "./monodepth2_data/final_datasets_paper/ground_truth/" + data[0] + "/" + data[1] + "/" + data[2][:-4] + "_disp.jpeg"
						imgd = cv2.imread(depth_file1,0)
						imgd = cv2.resize(imgd, (img_default.shape[1],img_default.shape[0]),interpolation=cv2.INTER_AREA)
						
						depth_file2 = depth_folders[0] + data[0] + "/" + data[1] + "/" + data2[2][:-4] + "_disp.jpeg"
						#print(depth_file1)
						#print(depth_file2)
						imgd2 = cv2.imread(depth_file2,0)
						imgd2 = cv2.resize(imgd2, (img_default.shape[1],img_default.shape[0]),interpolation=cv2.INTER_AREA)
						
						for child2 in child_1:
							dist1 = 100000000000
							indmin = -1
							mindepth1 = 1000000000
							mindepth2 = 1000000000
							
							
							mindepth1b = 1000000000
							mindepth2b = 1000000000
							for ind in range(len(child)):
								if used_obj[ind] == 0:
									x11  = float(child2.attrib['xtl'])
									x12  = float(child[ind].attrib['xtl'])
									
									x21  = float(child2.attrib['xbr'])
									x22  = float(child[ind].attrib['xbr'])
									
									y11  = float(child2.attrib['ytl'])
									y12  = float(child[ind].attrib['ytl'])
									
									y21  = float(child2.attrib['ybr'])
									y22  = float(child[ind].attrib['ybr'])
									if x11 <0:
										x11 = 0
									if x12 <0:
										x12 = 0
									if y11 <0:
										t11 = 0
									if y12 <0:
										y12 = 0
									
									x11 = math.ceil(x11)
									x21 = math.floor(x21)
									y11 = math.ceil(y11)
									y21 = math.floor(y21)
									x12 = math.ceil(x12)
									x22 = math.floor(x22)
									y12 = math.ceil(y12)
									y22 = math.floor(y22)
				
									if x21 >= img_default.shape[1]:
										x21 = img_default.shape[1]-1
										
									if x22 >= img_default.shape[1]:
										x22 = img_default.shape[1]-1
									if y21 >= img_default.shape[0]:
										y21 = img_default.shape[0]-1
									if y22 >= img_default.shape[0]:
										y22 = img_default.shape[0]-1
									
									#print(x11,x21,y11,y21)
									#print(x12,x22,y12,y22)
									
									#print(img.shape)
									dist_new = (x12-x11)*(x12-x11)+(x21-x22)*(x21-x22)+(y12-y11)*(y12-y11)+(y21-y22)*(y21-y22)
									if dist_new < dist1:
										dist1 = dist_new
										indmin = ind
										avg_depth_1 = 0
										avg_depth_2 = 0
										
										for indx1 in range(math.floor(x21-x11)):
											for indy1 in range(math.floor(y21-y11)):
												avg_depth_1 += imgd[indy1+math.floor(y11)][indx1+math.floor(x11)]
										for indx1 in range(math.floor(x22-x12)):
											for indy1 in range(math.floor(y22-y12)):
												avg_depth_2 += imgd[indy1+math.floor(y12)][indx1+math.floor(x12)]
										if math.floor(x21-x11)>0 and math.floor(x22-x12)>0 and math.floor(y21-y11)>0 and math.floor(y22-y12)>0:
											mindepth1 = avg_depth_1/(math.floor(x21-x11)*math.floor(y21-y11))
											mindepth2 = avg_depth_2/(math.floor(x22-x12)*math.floor(y22-y12))
											
										avg_depth_1b = 0
										avg_depth_2b = 0
										
										for indx1 in range(math.floor(x21-x11)):
											for indy1 in range(math.floor(y21-y11)):
												avg_depth_1b += imgd2[indy1+math.floor(y11)][indx1+math.floor(x11)]
										for indx1 in range(math.floor(x22-x12)):
											for indy1 in range(math.floor(y22-y12)):
												avg_depth_2b += imgd2[indy1+math.floor(y12)][indx1+math.floor(x12)]
										if math.floor(x21-x11)>0 and math.floor(x22-x12)>0 and math.floor(y21-y11)>0 and math.floor(y22-y12)>0:
											mindepth1b = avg_depth_1b/(math.floor(x21-x11)*math.floor(y21-y11))
											mindepth2b = avg_depth_2b/(math.floor(x22-x12)*math.floor(y22-y12))
											
							if indmin > -1:
								used_obj[indmin] = 1
								scor1[0] = scor1[0] + dist1
								scor1[daytime_map[data[0]]] = scor1[daytime_map[data[0]]] + dist1
								numbers1[0] = numbers1[0] + 1
								numbers1[daytime_map[data[0]]] = numbers1[daytime_map[data[0]]] + 1 
								
								scor1_depth[0] = scor1_depth[0] + (mindepth1-mindepth2)*(mindepth1-mindepth2)
								scor1_depth[daytime_map[data[0]]] = scor1_depth[daytime_map[data[0]]] + (mindepth1-mindepth2)*(mindepth1-mindepth2)
								
								if mindepth1 < 1000000000:
									numbers1_depth[0] = numbers1_depth[0] + 1
									numbers1_depth[daytime_map[data[0]]] = numbers1_depth[daytime_map[data[0]]] + 1

								scor1_depth_predicted[0] = scor1_depth_predicted[0] + (mindepth1b-mindepth2b)*(mindepth1b-mindepth2b)
								scor1_depth_predicted[daytime_map[data[0]]] = scor1_depth_predicted[daytime_map[data[0]]] + (mindepth1b-mindepth2b)*(mindepth1b-mindepth2b)
								
								if mindepth1b < 1000000000:
									numbers1_depth_predicted[0] = numbers1_depth_predicted[0] + 1
									numbers1_depth_predicted[daytime_map[data[0]]] = numbers1_depth_predicted[daytime_map[data[0]]] + 1
								
								#print(mindepth1, mindepth2)
								#print(numbers1_depth[0])
								
								#print(scor1_depth[0])
								#print( (scor1_depth[0]/numbers1_depth[0])**0.5)
						
						segm_file = ""
						for y in ord1[data[0]][data[1]]:
							if ord1[data[0]][data[1]][y] ==1:
								
								mindepth1 = 1000000000
								mindepth2 = 1000000000
								
								mindepth1b = 1000000000
								mindepth2b = 1000000000
								nr = y.split("-")[0]
								nr2 = int(nr)-1
								segm_file = "./segm_data_fcn/" + data[0]  + "/" + data[1]  + "/" +str(nr2)+".png"
								

								#print(segm_file2)
								
								segm_file_fcn = "./segm_data_fcn/" + data[0]  + "/" + data[1]  + "/" +str(nr2) + "-frame_fcn.png"
								predicted = raws_folders[0] + data2[0]+ "/" + data2[1] + "/" + data2[2]
								numeimg = './raw_data/prednet_small/' + path2
								#print(numeimg)
								img_default = cv2.imread(numeimg, 0)
								
								img0 = cv2.imread(predicted,0)
								img0 = cv2.resize(img0, (img_default.shape[1],img_default.shape[0]),interpolation=cv2.INTER_AREA)
								img = cv2.imread(segm_file,0)
								img = cv2.resize(img, (img0.shape[1],img0.shape[0]),interpolation=cv2.INTER_AREA)
								#print(img0.shape)
								#print(img.shape)
								
								rows,cols = img.shape
								min_w = 10000
								max_w = -1
								min_h = 10000
								max_h = -1
								for i in range(rows):
									for j in range(cols):
										if img[i, j] == 255 and i > max_w:
											max_w = i
										if img[i, j] == 255 and i < min_w:
											min_w = i
										if img[i, j] == 255 and j > max_h:
											max_h = j
										if img[i, j] == 255 and j < min_h:
											min_h = j
								
								segm_file2 = "./segm_data_fcn/" + data[0]  + "/" + data[1]  + "/" + data[2][:-10] +".png"
								print(segm_file2)
								img3 = cv2.imread(segm_file2,0)
								if img3 is None:
									continue
								img3 = cv2.resize(img3, (img0.shape[1],img0.shape[0]),interpolation=cv2.INTER_AREA)
								
								min_w3 = 10000
								max_w3 = -1
								min_h3 = 10000
								max_h3 = -1
								for i in range(rows):
									for j in range(cols):
										if img3[i, j] == 255 and i > max_w3:
											max_w3 = i
										if img3[i, j] == 255 and i < min_w3:
											min_w3 = i
										if img3[i, j] == 255 and j > max_h3:
											max_h3 = j
										if img3[i, j] == 255 and j < min_h3:
											min_h3 = j
											
								
															
								#print(segm_file)
								#print(segm_file_fcn)
								#print(min_w,max_w,min_h,max_h)
								img2 = cv2.imread(segm_file_fcn,0)
								img2 = cv2.resize(img2, (img0.shape[1],img0.shape[0]),interpolation=cv2.INTER_AREA)
								colors = []
								for ind in range(256):
									colors.append(0)
								
								for i in range(rows):
									for j in range(cols):
										if  img[i,j] == 255:
											#print(img2[i,j])
											colors[img2[i,j]]= colors[img2[i,j]]+1
								indmax = 0
								for i in range(256):
									if colors[i] > colors[indmax]:
										indmax = i
								colmax = indmax
								#print(colmax)
								min_w2 = 10000
								max_w2 = -1
								min_h2 = 10000
								max_h2 = -1
								
								for i in range(rows):
									for j in range(cols):
										if img2[i, j] == colmax and i > max_w2:
											max_w2 = i
										if img2[i, j] == colmax and i < min_w2:
											min_w2 = i
										if img2[i, j] == colmax and j > max_h2:
											max_h2 = j
										if img2[i, j] == colmax and j < min_h2:
											min_h2 = j
											
								
								
								segm_file_fcn2 = segm_folders[0] + data[0] + "/" + data[1] + "/" + data2[2][:-4] + "_fcn.png"
								img4 = cv2.imread(segm_file_fcn2,0)
								img4 = cv2.resize(img4, (img0.shape[1],img0.shape[0]),interpolation=cv2.INTER_AREA)
								colors = []
								for ind in range(256):
									colors.append(0)
								
								for i in range(rows):
									for j in range(cols):
										if  img3[i,j] == 255:
											#print(img2[i,j])
											colors[img4[i,j]]= colors[img4[i,j]]+1
								indmax = 0
								for i in range(256):
									if colors[i] > colors[indmax]:
										indmax = i
								colmax = indmax
								#print(colmax)
								min_w4 = 10000
								max_w4 = -1
								min_h4 = 10000
								max_h4 = -1
								
								for i in range(rows):
									for j in range(cols):
										if img4[i, j] == colmax and i > max_w4:
											max_w4 = i
										if img4[i, j] == colmax and i < min_w4:
											min_w4 = i
										if img4[i, j] == colmax and j > max_h4:
											max_h4 = j
										if img4[i, j] == colmax and j < min_h4:
											min_h4 = j
											
												
								#print(min_w2,max_w2,min_h2,max_h2)
											
								used_obj_segm = []
								for ii in range(10):
									used_obj_segm.append([])								
									for child2 in child:
										used_obj_segm[ii].append(0)
								#print(len(used_obj_segm))
								#print(path2)
								#print("!!")
								
								for ii in range(10):	
									for child2 in child_1:
										dist1 = 100000000000
										indmin = -1
										size_class = -1
										
										for ind in range(len(child)):
											if used_obj_segm[ii][ind] == 0:
												
														
												x11  = float(child2.attrib['xtl'])
												x12  = float(child[ind].attrib['xtl'])
												
												x21  = float(child2.attrib['xbr'])
												x22  = float(child[ind].attrib['xbr'])
												
												y11  = float(child2.attrib['ytl'])
												y12  = float(child[ind].attrib['ytl'])
												
												y21  = float(child2.attrib['ybr'])
												y22  = float(child[ind].attrib['ybr'])
												
												if x11 <0:
													x11 = 0
												if x12 <0:
													x12 = 0
												if y11 <0:
													t11 = 0
												if y12 <0:
													y12 = 0
												if x21 >= img_default.shape[1]:
													x21 = img_default.shape[1]-1
													
												if x22 >= img_default.shape[1]:
													x22 = img_default.shape[1]-1
												if y21 >= img_default.shape[0]:
													y21 = img_default.shape[0]-1
												if y22 >= img_default.shape[0]:
													y22 = img_default.shape[0]-1
												xo1 = 0
												xo2 = 0
												yo1 = 0
												yo2 = 0
												dmin1 = 1000000000
												for child_12 in root2:
													if 'name' in child_12.attrib:
														data21 = child_12.attrib['name'].split("_")
														if data21[0] == data[0] and data21[1] == data[1]:
															if int(data21[2][0])-4 == 1:
																#print(data21)
																#print(data2)
																
																for child22 in child_12:
																	x112 = float(child22.attrib['xtl'])
																	x212 = float(child22.attrib['xbr'])
																	y112 = float(child22.attrib['ytl'])
																	y212 = float(child22.attrib['ybr'])
																	if x112 <0:
																		x112 = 0
																	if y112 <0:
																		y112 = 0
																	if x212   >= img_default.shape[1]:
																		x212 = img_default.shape[1]-1
																	if y212   >= img_default.shape[0]:
																		y212 = img_default.shape[0]-1
													
																	dist_new = (x112-x11)*(x112-x11)+(x212-x21)*(x212-x21)+(y112-y11)*(y112-y11)+(y212-y21)*(y212-y21)
																	if dist_new < dmin1:
																		dmin1 = dist_new
																		xo1 = x112
																		yo1 = y112
																		xo2 = x212
																		yo2 = y212
												#print (x11,x21,y11,y21)
												#print (xo1,xo2,yo1,yo2)
													
												procx11 = (x11-min_w4)/ (max_w4-min_w4)
												procx21 = (x21-min_w4)/ (max_w4-min_w4)
												procy11 = (y11-min_h4)/ (max_h4-min_h4)
												procy21 = (y21-min_h4)/ (max_h4-min_h4)
													
												procxo1 = (xo1-min_w2)/ (max_w2-min_w2)
												procxo2 = (xo2-min_w2)/ (max_w2-min_w2)
												procyo1 = (yo1-min_h2)/ (max_h2-min_h2)
												procyo2 = (yo2-min_h2)/ (max_h2-min_h2)
													
												x11_new = (procx11*(ii+1)+procxo1*(10-ii-1))/10 * (max_w4-min_w4) + min_w4
													
												x21_new = (procx21*(ii+1)+procxo2*(10-ii-1))/10 * (max_w4-min_w4) + min_w4
													
												y11_new = (procy11*(ii+1)+procyo1*(10-ii-1))/10 * (max_h4-min_h4) + min_h4
													
												y21_new = (procy21*(ii+1)+procyo2*(10-ii-1))/10 * (max_h4-min_h4) + min_h4
												
												if x11_new <0:
													x11_new = 0
												if y11_new <0:
													y11_new = 0
												if x21_new  >= img_default.shape[1]:
													x21_new = img_default.shape[1]-1
												if y21_new   >= img_default.shape[0]:
													y21_new = img_default.shape[0]-1
#												if ii == 9:
#													print(x11_new,x21_new,y11_new,y21_new)
#													print(x12,x22,y12,y22)
												dist_new = (x12-x11_new)*(x12-x11_new)+(x22-x21_new)*(x22-x21_new)+(y12-y11_new)*(y12-y11_new)+(y22-y21_new)*(y22-y21_new)
												if dist_new < dist1:
													dist1 = dist_new
													
													indmin = ind
													avg_depth_1 = 0
													avg_depth_2 = 0
													avg_depth_1b = 0
													avg_depth_2b = 0
													
													dimobj = (x22-x12)*(y22-y12)
													for jj in range(len(classes)-1):
														if dimobj >= classes[jj] and dimobj < classes[jj+1]:
															size_class = jj
															break
													
													for indx1 in range(math.floor(x21_new-x11_new)):
														for indy1 in range(math.floor(y21_new-y11_new)):
															avg_depth_1 += imgd[indy1+math.floor(y11_new)][indx1+math.floor(x11_new)]
															avg_depth_1b += imgd2[indy1+math.floor(y11_new)][indx1+math.floor(x11_new)]
															
													for indx1 in range(math.floor(x22-x12)):
														for indy1 in range(math.floor(y22-y12)):
															avg_depth_2 += imgd[indy1+math.floor(y12)][indx1+math.floor(x12)]
															avg_depth_2b += imgd2[indy1+math.floor(y12)][indx1+math.floor(x12)]
															
													if math.floor(x21_new-x11_new)>0 and math.floor(x22-x12)>0 and math.floor(y21_new-y11_new)>0 and math.floor(y22-y12)>0:
														mindepth1 = avg_depth_1/(math.floor(x21_new-x11_new)*math.floor(y21_new-y11_new))
														mindepth2 = avg_depth_2/(math.floor(x22-x12)*math.floor(y22-y12))
														
														mindepth1b = avg_depth_1b/(math.floor(x21_new-x11_new)*math.floor(y21_new-y11_new))
														mindepth2b = avg_depth_2b/(math.floor(x22-x12)*math.floor(y22-y12))
										if indmin > -1:
											used_obj_segm[ii][indmin] = 1
											scor1_segm[0][ii] = scor1_segm[0][ii] + dist1
											scor1_segm[daytime_map[data[0]]][ii] = scor1_segm[daytime_map[data[0]]][ii] + dist1
											
											numbers1_segm[0][ii] = numbers1_segm[0][ii] + 1
											numbers1_segm[daytime_map[data[0]]][ii] = numbers1_segm[daytime_map[data[0]]][ii] + 1
											
											scor1_segm_depth[0][ii] = scor1_segm_depth[0][ii] + (mindepth1-mindepth2)*(mindepth1-mindepth2)
											scor1_segm_depth[daytime_map[data[0]]][ii] = scor1_segm_depth[daytime_map[data[0]]][ii] + (mindepth1-mindepth2)*(mindepth1-mindepth2)
											
											numbers1_segm_depth[0][ii] = numbers1_segm_depth[0][ii] + 1
											numbers1_segm_depth[daytime_map[data[0]]][ii] = numbers1_segm_depth[daytime_map[data[0]]][ii] + 1
											
											scor1_segm_depth_predicted[0][ii] = scor1_segm_depth_predicted[0][ii] + (mindepth1b-mindepth2b)*(mindepth1b-mindepth2b)
											scor1_segm_depth_predicted[daytime_map[data[0]]][ii] = scor1_segm_depth_predicted[daytime_map[data[0]]][ii] + (mindepth1b-mindepth2b)*(mindepth1b-mindepth2b)
											numbers1_segm_depth_predicted[0][ii] = numbers1_segm_depth_predicted[0][ii] + 1
											numbers1_segm_depth_predicted[daytime_map[data[0]]][ii] = numbers1_segm_depth_predicted[daytime_map[data[0]]][ii] + 1
											
											scor1_segm_classes[size_class][ii] = scor1_segm_classes[size_class][ii] + dist1
											numbers1_segm_classes[size_class][ii] = numbers1_segm_classes[size_class][ii] + 1
											
								#			if ii == 9:
								#				print(mindepth1, mindepth2)
								#				print(numbers1_segm_depth[ii])
								#	
								#				print( (scor1_segm_depth[ii]))
								#				print( (scor1_segm_depth[ii]/numbers1_segm_depth[ii])**0.5)		
												
						
											 



f = open(predicted_json[0])
datas = json.load(f)



scor2 = [0,0,0,0]
numbers2 = [0,0,0,0]


scor2_depth = [0,0,0,0]
numbers2_depth = [0,0,0,0]

scor2_depth_predicted = [0,0,0,0]
numbers2_depth_predicted = [0,0,0,0]


scor2_segm =   [[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]
numbers2_segm = [[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]


scor2_segm_fcn =    [0,0,0,0,0,0,0,0,0,0,0]
numbers2_segm_fcn = [0,0,0,0,0,0,0,0,0,0,0]

scor2_segm_depth =    [[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]
numbers2_segm_depth = [[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]

scor2_segm_depth_predicted = [[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]
numbers2_segm_depth_predicted = [[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0]]

	
scor2_segm_classes = []
numbers2_segm_classes = []
for ii in range(len(classes)):
	scor2_segm_classes.append([0,0,0,0,0,0,0,0,0,0,0])
	numbers2_segm_classes.append( [0,0,0,0,0,0,0,0,0,0,0])


for child in root:
#	print(child.attrib)
	data = ""
	if 'name' in child.attrib:
		data = child.attrib['name'].split("_")
		path = data[0] + "/" + data[1] +"/" + data[2]
		#print(path)
		for x in datas:
			data2 = x['filename'].split("/")[1:]
			if data2[0] == data[0] and data2[1] == data[1]:
				if int(data2[2][0])-4 == ord1[data[0]][data[1]][data[2]]:
					#print(x['filename'])
					#print(x)
					numeimg = './raw_data/' + x['filename']
					img = cv2.imread(numeimg, 0)
					#print(img.shape)
					used_obj = []
					for child2 in child:
						used_obj.append(0)
					#print(len(used_obj))
					
					print(numeimg)
					img_default = cv2.imread(numeimg, 0)
								
					depth_file1 = "./monodepth2_data/final_datasets_paper/ground_truth/" + data[0] + "/" + data[1] + "/" + data[2][:-4] + "_disp.jpeg"
					imgd = cv2.imread(depth_file1,0)
					imgd = cv2.resize(img, (img_default.shape[1],img_default.shape[0]),interpolation=cv2.INTER_AREA)
						
					depth_file2 = depth_folders[0] + data[0] + "/" + data[1] + "/" + data2[2][:-4] + "_disp.jpeg"
					imgd2 = cv2.imread(depth_file2,0)
					imgd2 = cv2.resize(imgd2, (img_default.shape[1],img_default.shape[0]),interpolation=cv2.INTER_AREA)
						
						
					for y in x['objects']:
						#print(y) #do shit with generated labels
						dist1 = 100000000000
						indmin = -1
						mindepth1 = 1000000000
						mindepth2 = 1000000000
						mindepth1b = 1000000000
						mindepth2b = 1000000000
						for ind in range(len(child)):
							if used_obj[ind] == 0:
								x11  = float(y['relative_coordinates']['center_x'])-float(y['relative_coordinates']['width'])
								x11 = x11*img.shape[1]
								x12  = float(child[ind].attrib['xtl'])
								
								x21  = float(y['relative_coordinates']['center_x'])+float(y['relative_coordinates']['width'])
								x21 = x21*img.shape[1]
								x22  = float(child[ind].attrib['xbr'])
								
								
								y11  = float(y['relative_coordinates']['center_y'])-float(y['relative_coordinates']['height'])
								y11 = y11 * img.shape[0]
								y12  = float(child[ind].attrib['ytl'])
								
								y21  = float(y['relative_coordinates']['center_y'])+float(y['relative_coordinates']['height'])
								y21 = y21 * img.shape[0]
								y22  = float(child[ind].attrib['ybr'])
								
								x11 = math.ceil(x11)
								x21 = math.floor(x21)
								y11 = math.ceil(y11)
								y21 = math.floor(y21)
								x12 = math.ceil(x12)
								x22 = math.floor(x22)
								y12 = math.ceil(y12)
								y22 = math.floor(y22)
				
								if x11 <0:
									x11 = 0
								if x12 <0:
									x12 = 0
								if y11 <0:
									t11 = 0
								if y12 <0:
									y12 = 0
								if x21 >= img_default.shape[1]:
									x21 = img_default.shape[1]-1
									
								if x22 >= img_default.shape[1]:
									x22 = img_default.shape[1]-1
								if y21 >= img_default.shape[0]:
									y21 = img_default.shape[0]-1
								if y22 >= img_default.shape[0]:
									y22 = img_default.shape[0]-1
								
								#print(x11,x12,y11,y12)
								#print(x21,x22,y21,y22)
								#print(img.shape)
								dist_new = (x12-x11)*(x12-x11)+(x21-x22)*(x21-x22)+(y12-y11)*(y12-y11)+(y21-y22)*(y21-y22)
								if dist_new < dist1:
									dist1 = dist_new
									indmin = ind
									avg_depth_1 = 0
									avg_depth_2 = 0
									
									for indx1 in range(math.floor(x21-x11)):
										for indy1 in range(math.floor(y21-y11)):
											avg_depth_1 += imgd[indy1+math.floor(y11)][indx1+math.floor(x11)]
									for indx1 in range(math.floor(x22-x12)):
										for indy1 in range(math.floor(y22-y12)):
											avg_depth_2 += imgd[indy1+math.floor(y12)][indx1+math.floor(x12)]
									if math.floor(x21-x11)>0 and math.floor(x22-x12)>0 and math.floor(y21-y11)>0 and math.floor(y22-y12)>0:
										mindepth1 = avg_depth_1/(math.floor(x21-x11)*math.floor(y21-y11))
										mindepth2 = avg_depth_2/(math.floor(x22-x12)*math.floor(y22-y12))
										
									avg_depth_1b = 0
									avg_depth_2b = 0
									
									for indx1 in range(math.floor(x21-x11)):
										for indy1 in range(math.floor(y21-y11)):
											avg_depth_1b += imgd2[indy1+math.floor(y11)][indx1+math.floor(x11)]
									for indx1 in range(math.floor(x22-x12)):
										for indy1 in range(math.floor(y22-y12)):
											avg_depth_2b += imgd2[indy1+math.floor(y12)][indx1+math.floor(x12)]
									if math.floor(x21-x11)>0 and math.floor(x22-x12)>0 and math.floor(y21-y11)>0 and math.floor(y22-y12)>0:
										mindepth1b = avg_depth_1b/(math.floor(x21-x11)*math.floor(y21-y11))
										mindepth2b = avg_depth_2b/(math.floor(x22-x12)*math.floor(y22-y12))
											
						if indmin > -1:
							used_obj[indmin] = 1
							scor2[0] = scor2[0] + dist1
							scor2[daytime_map[data[0]]] = scor2[daytime_map[data[0]]] + dist1
							numbers2[0] = numbers2[0] + 1
							numbers2[daytime_map[data[0]]] = numbers2[daytime_map[data[0]]] + 1 	
								
							scor2_depth[0] = scor2_depth[0] + (mindepth1-mindepth2)*(mindepth1-mindepth2)
							scor2_depth[daytime_map[data[0]]] = scor2_depth[daytime_map[data[0]]] + (mindepth1-mindepth2)*(mindepth1-mindepth2)
							if mindepth1 < 1000000000:
								numbers2_depth[0] = numbers2_depth[0] + 1
								numbers2_depth[daytime_map[data[0]]] = numbers2_depth[daytime_map[data[0]]] + 1
							
							scor2_depth_predicted[0] = scor2_depth_predicted[0] + (mindepth1b-mindepth2b)*(mindepth1b-mindepth2b)
							scor2_depth_predicted[daytime_map[data[0]]] = scor2_depth_predicted[daytime_map[data[0]]] + (mindepth1b-mindepth2b)*(mindepth1b-mindepth2b)
							if mindepth1b < 1000000000:
								numbers2_depth_predicted[0] = numbers2_depth_predicted[0] + 1
								numbers2_depth_predicted[daytime_map[data[0]]] = numbers2_depth_predicted[daytime_map[data[0]]] + 1
								
								
					
					
					
					for yname in ord1[data[0]][data[1]]:
						if ord1[data[0]][data[1]][yname] ==1:
								
							mindepth1 = 1000000000
							mindepth2 = 1000000000
							
							
							mindepth1b = 1000000000
							mindepth2b = 1000000000
							
							nr = yname.split("-")[0]
							nr2 = int(nr)-1
							segm_file = "./segm_data_fcn/" + data[0]  + "/" + data[1]  + "/" +str(nr2)+".png"
							segm_file_fcn = "./segm_data_fcn/" + data[0]  + "/" + data[1]  + "/" +str(nr2) + "-frame_fcn.png"
							predicted = raws_folders[0] + data2[0]+ "/" + data2[1] + "/" + data2[2]
							numeimg = './raw_data/' + x['filename']
							#print(numeimg)
							img_default = cv2.imread(numeimg, 0)
							
							img0 = cv2.imread(predicted,0)
							img0 = cv2.resize(img0, (img_default.shape[1],img_default.shape[0]),interpolation=cv2.INTER_AREA)
							img = cv2.imread(segm_file,0)
							img = cv2.resize(img, (img0.shape[1],img0.shape[0]),interpolation=cv2.INTER_AREA)
							#print(img0.shape)
							#print(img.shape)
									
							rows,cols = img.shape
							min_w = 10000
							max_w = -1
							min_h = 10000
							max_h = -1
							for i in range(rows):
								for j in range(cols):
									if img[i, j] == 255 and i > max_w:
										max_w = i
									if img[i, j] == 255 and i < min_w:
										min_w = i
									if img[i, j] == 255 and j > max_h:
										max_h = j
									if img[i, j] == 255 and j < min_h:
										min_h = j
							
											
							#print(segm_file)
							#print(segm_file_fcn)
							#print(min_w,max_w,min_h,max_h)
							img2 = cv2.imread(segm_file_fcn,0)	
							img2 = cv2.resize(img2, (img0.shape[1],img0.shape[0]),interpolation=cv2.INTER_AREA)
							colors = []
							
							
							for ind in range(256):
								colors.append(0)
							
							for i in range(rows):
								for j in range(cols):
									if  img[i,j] == 255:
										#print(img2[i,j])
										colors[img2[i,j]]= colors[img2[i,j]]+1
							indmax = 0
							for i in range(256):
								if colors[i] > colors[indmax]:
									indmax = i
							colmax = indmax
							#print(colmax)
							min_w2 = 10000
							max_w2 = -1
							min_h2 = 10000
							max_h2 = -1
								
							for i in range(rows):
								for j in range(cols):
									if img2[i, j] == colmax and i > max_w2:
										max_w2 = i
									if img2[i, j] == colmax and i < min_w2:
										min_w2 = i
									if img2[i, j] == colmax and j > max_h2:
										max_h2 = j
									if img2[i, j] == colmax and j < min_h2:
										min_h2 = j
							
							segm_file2 = "./segm_data_fcn/" + data[0]  + "/" + data[1]  + "/" + data[2][:-10] +".png"
							img3 = cv2.imread(segm_file2,0)
							if img3 is None:
								continue
							img3 = cv2.resize(img3, (img0.shape[1],img0.shape[0]),interpolation=cv2.INTER_AREA)
									
							min_w3 = 10000
							max_w3 = -1
							min_h3 = 10000
							max_h3 = -1
							for i in range(rows):
								for j in range(cols):
									if img3[i, j] == 255 and i > max_w3:
										max_w3 = i
									if img3[i, j] == 255 and i < min_w3:
										min_w3 = i
									if img3[i, j] == 255 and j > max_h3:
										max_h3 = j
									if img3[i, j] == 255 and j < min_h3:
										min_h3 = j	
							#print(min_w2,max_w2,min_h2,max_h2)
							
							
							segm_file_fcn2 = segm_folders[0] + data[0] + "/" + data[1] + "/" + data2[2][:-4] + "_fcn.png"
							print(segm_file_fcn2)
							img4 = cv2.imread(segm_file_fcn2,0)
							img4 = cv2.resize(img4, (img0.shape[1],img0.shape[0]),interpolation=cv2.INTER_AREA)
							colors = []
							for ind in range(256):
								colors.append(0)
								
							for i in range(rows):
								for j in range(cols):
									if  img3[i,j] == 255:
										#print(img2[i,j])
										colors[img4[i,j]]= colors[img4[i,j]]+1
							indmax = 0
							for i in range(256):
								if colors[i] > colors[indmax]:
									indmax = i
							colmax = indmax
							#print(colmax)
							min_w4 = 10000
							max_w4 = -1
							min_h4 = 10000
							max_h4 = -1
								
							for i in range(rows):
								for j in range(cols):
									if img4[i, j] == colmax and i > max_w4:
										max_w4 = i
									if img4[i, j] == colmax and i < min_w4:
										min_w4 = i
									if img4[i, j] == colmax and j > max_h4:
										max_h4 = j
									if img4[i, j] == colmax and j < min_h4:
										min_h4 = j
								
							used_obj_segm = []
							for ii in range(10):
								used_obj_segm.append([])								
								for child2 in child:
									used_obj_segm[ii].append(0)
							#print(len(used_obj_segm))
							#print(path2)
							#print("!!")
							
							for ii in range(10):	
								for y in x['objects']:
									dist1 = 100000000000
									indmin = -1
									size_class = -1
									
									for ind in range(len(child)):
										if used_obj_segm[ii][ind] == 0:
											
													
											x11  = float(y['relative_coordinates']['center_x'])-float(y['relative_coordinates']['width'])
											x11 = x11*img.shape[1]
											x12  = float(child[ind].attrib['xtl'])
											
											x21  = float(y['relative_coordinates']['center_x'])+float(y['relative_coordinates']['width'])
											x21 = x21*img.shape[1]
											x22  = float(child[ind].attrib['xbr'])
											
											
											y11  = float(y['relative_coordinates']['center_y'])-float(y['relative_coordinates']['height'])
											y11 = y11 * img.shape[0]
											y12  = float(child[ind].attrib['ytl'])
											
											y21  = float(y['relative_coordinates']['center_y'])+float(y['relative_coordinates']['height'])
											y21 = y21 * img.shape[0]
											y22  = float(child[ind].attrib['ybr'])
											if x11 <0:
												x11 = 0
											if x12 <0:
												x12 = 0
											if y11 <0:
												t11 = 0
											if y12 <0:
												y12 = 0
											if x21 >= img_default.shape[1]:
												x21 = img_default.shape[1]-1
												
											if x22 >= img_default.shape[1]:
												x22 = img_default.shape[1]-1
											if y21 >= img_default.shape[0]:
												y21 = img_default.shape[0]-1
											if y22 >= img_default.shape[0]:
												y22 = img_default.shape[0]-1
											xo1 = 0
											xo2 = 0
											yo1 = 0
											yo2 = 0
											#print(x11,x12,y11,y12)
											dmin1 = 1000000000
											for x_json in datas:
												data2_json = x_json['filename'].split("/")[1:]
												#print(data2_json)
												if data2_json[0] == data[0] and data2_json[1] == data[1]:
													if int(data2_json[2][0])-4 == 1:
														
														for y_json in x_json['objects']:
															x112  = float(y_json['relative_coordinates']['center_x'])-float(y_json['relative_coordinates']['width'])
															x112 = x112*img.shape[1]
															
															x212  = float(y_json['relative_coordinates']['center_x'])+float(y_json['relative_coordinates']['width'])
															x212 = x212*img.shape[1]
															
															
															y112  = float(y_json['relative_coordinates']['center_y'])-float(y_json['relative_coordinates']['height'])
															y112 = y112 * img.shape[0]
															
															y212  = float(y_json['relative_coordinates']['center_y'])+float(y_json['relative_coordinates']['height'])
															y212 = y212 * img.shape[0]
															
															if x112 <0:
																x112 = 0
															if y112 <0:
																y112 = 0
															if x212   >= img_default.shape[1]:
																x212 = img_default.shape[1]-1
															if y212   >= img_default.shape[0]:
																y212 = img_default.shape[0]-1
															
															dist_new = (x112-x11)*(x112-x11)+(x212-x21)*(x212-x21)+(y112-y11)*(y112-y11)+(y212-y21)*(y212-y21)
															if dist_new < dmin1:
																dmin1 = dist_new
																xo1 = x112
																yo1 = y112
																xo2 = x212
																yo2 = y212
											#print (x11,x21,y11,y21)
											#print (xo1,xo2,yo1,yo2)
											
											procx11 = (x11-min_w4)/ (max_w4-min_w4)
											procx21 = (x21-min_w4)/ (max_w4-min_w4)
											procy11 = (y11-min_h4)/ (max_h4-min_h4)
											procy21 = (y21-min_h4)/ (max_h4-min_h4)
												
											procxo1 = (xo1-min_w2)/ (max_w2-min_w2)
											procxo2 = (xo2-min_w2)/ (max_w2-min_w2)
											procyo1 = (yo1-min_h2)/ (max_h2-min_h2)
											procyo2 = (yo2-min_h2)/ (max_h2-min_h2)
												
											x11_new = (procx11*(ii+1)+procxo1*(10-ii-1))/10 * (max_w4-min_w4) + min_w4
												
											x21_new = (procx21*(ii+1)+procxo2*(10-ii-1))/10 * (max_w4-min_w4) + min_w4
												
											y11_new = (procy11*(ii+1)+procyo1*(10-ii-1))/10 * (max_h4-min_h4) + min_h4
												
											y21_new = (procy21*(ii+1)+procyo2*(10-ii-1))/10 * (max_h4-min_h4) + min_h4
												
											
											x11_new = math.floor(x11_new)
											x21_new = math.ceil(x21_new)
											y11_new = math.floor(y11_new)
											y21_new = math.ceil(y21_new)
											x12 = math.floor(x12)
											x22 = math.ceil(x22)
											y12 = math.floor(y12)
											y22 = math.ceil(y22) 
					
											if x11_new <0:
												x11_new = 0
											if y11_new <0:
												y11_new = 0
											if x21_new  >= img_default.shape[1]:
												x21_new = img_default.shape[1]-1
											if y21_new   >= img_default.shape[0]:
												y21_new = img_default.shape[0]-1
											#print(x11_new,x21_new,y11_new,y21_new)
												#print(ii)	
											dist_new = (x12-x11_new)*(x12-x11_new)+(x22-x21_new)*(x22-x21_new)+(y12-y11_new)*(y12-y11_new)+(y22-y21_new)*(y22-y21_new)
											if dist_new < dist1:
												dist1 = dist_new
												indmin = ind
												avg_depth_1 = 0
												avg_depth_2 = 0
												avg_depth_1b = 0
												avg_depth_2b = 0
												
												dimobj = (x22-x12)*(y22-y12)
												for jj in range(len(classes)-1):
													if dimobj >= classes[jj] and dimobj < classes[jj+1]:
														size_class = jj
														break
												for indx1 in range(math.floor(x21_new-x11_new)):
													for indy1 in range(math.floor(y21_new-y11_new)):
														avg_depth_1 += imgd[indy1+math.floor(y11_new)][indx1+math.floor(x11_new)]
														avg_depth_1b += imgd2[indy1+math.floor(y11_new)][indx1+math.floor(x11_new)]
															
												for indx1 in range(math.floor(x22-x12)):
													for indy1 in range(math.floor(y22-y12)):
														avg_depth_2 += imgd[indy1+math.floor(y12)][indx1+math.floor(x12)]
														avg_depth_2b += imgd2[indy1+math.floor(y12)][indx1+math.floor(x12)]
														
												if math.floor(x21_new-x11_new)>0 and math.floor(x22-x12)>0 and math.floor(y21_new-y11_new)>0 and math.floor(y22-y12)>0:
													mindepth1 = avg_depth_1/(math.floor(x21_new-x11_new)*math.floor(y21_new-y11_new))
													mindepth2 = avg_depth_2/(math.floor(x22-x12)*math.floor(y22-y12))
													
													mindepth1b = avg_depth_1b/(math.floor(x21_new-x11_new)*math.floor(y21_new-y11_new))
													mindepth2b = avg_depth_2b/(math.floor(x22-x12)*math.floor(y22-y12))
									if indmin > -1:
										used_obj_segm[ii][indmin] = 1
										scor2_segm[0][ii] = scor2_segm[0][ii] + dist1
										scor2_segm[daytime_map[data[0]]][ii] = scor2_segm[daytime_map[data[0]]][ii] + dist1
											
										numbers2_segm[0][ii] = numbers2_segm[0][ii] + 1
										numbers2_segm[daytime_map[data[0]]][ii] = numbers2_segm[daytime_map[data[0]]][ii] + 1
											
										scor2_segm_depth[0][ii] = scor2_segm_depth[0][ii] + (mindepth1-mindepth2)*(mindepth1-mindepth2)
										scor2_segm_depth[daytime_map[data[0]]][ii] = scor2_segm_depth[daytime_map[data[0]]][ii] + (mindepth1-mindepth2)*(mindepth1-mindepth2)
											
										numbers2_segm_depth[0][ii] = numbers2_segm_depth[0][ii] + 1
										numbers2_segm_depth[daytime_map[data[0]]][ii] = numbers2_segm_depth[daytime_map[data[0]]][ii] + 1
											
										scor2_segm_depth_predicted[0][ii] = scor2_segm_depth_predicted[0][ii] + (mindepth1b-mindepth2b)*(mindepth1b-mindepth2b)
										scor2_segm_depth_predicted[daytime_map[data[0]]][ii] = scor2_segm_depth_predicted[daytime_map[data[0]]][ii] + (mindepth1b-mindepth2b)*(mindepth1b-mindepth2b)
										numbers2_segm_depth_predicted[0][ii] = numbers2_segm_depth_predicted[0][ii] + 1
										numbers2_segm_depth_predicted[daytime_map[data[0]]][ii] = numbers2_segm_depth_predicted[daytime_map[data[0]]][ii] + 1
											
										scor2_segm_classes[size_class][ii] = scor2_segm_classes[size_class][ii] + dist1
										numbers2_segm_classes[size_class][ii] = numbers2_segm_classes[size_class][ii] + 1


								
print('scor fara segm')
print('avg:')
print((scor1[0]/numbers1[0])**	0.5)
print('day:')
print((scor1[1]/numbers1[1])**	0.5)
print('dusk:')
print((scor1[2]/numbers1[2])**	0.5)
print('night:')
print((scor1[3]/numbers1[3])**	0.5)



print('scor depth fara segm')
print('avg:')
print((scor1_depth[0]/numbers1_depth[0])**0.5)
print('day:')
print((scor1_depth[1]/numbers1_depth[1])**0.5)
print('dusk:')
print((scor1_depth[2]/numbers1_depth[2])**0.5)
print('night:')
print((scor1_depth[3]/numbers1_depth[3])**0.5)



print('scor depth predicted fara segm')
print('avg:')
print((scor1_depth_predicted[0]/numbers1_depth_predicted[0])**0.5)
print('day:')
print((scor1_depth_predicted[1]/numbers1_depth_predicted[1])**0.5)
print('dusk:')
print((scor1_depth_predicted[2]/numbers1_depth_predicted[2])**0.5)
print('night:')
print((scor1_depth_predicted[3]/numbers1_depth_predicted[3])**0.5)


print('scor segm')
print('avg:')
minds = 1000000000
for ii in range(10):
	if (scor1_segm[0][ii]/numbers1_segm[0][ii])**0.5 < minds:
		minds = (scor1_segm[0][ii]/numbers1_segm[0][ii])**0.5
print(minds)
print('day:')
minds = 1000000000
for ii in range(10):
	if (scor1_segm[1][ii]/numbers1_segm[1][ii])**0.5 < minds:
		minds = (scor1_segm[1][ii]/numbers1_segm[1][ii])**0.5
print(minds)

print('dusk:')
minds = 1000000000
for ii in range(10):
	if (scor1_segm[2][ii]/numbers1_segm[2][ii])**0.5 < minds:
		minds = (scor1_segm[2][ii]/numbers1_segm[2][ii])**0.5
print(minds)
print('night:')
minds = 1000000000
for ii in range(10):
	if (scor1_segm[3][ii]/numbers1_segm[3][ii])**0.5 < minds:
		minds = (scor1_segm[3][ii]/numbers1_segm[3][ii])**0.5
print(minds)


print('scor depth segm')
print('avg:')
minds = 1000000000
for ii in range(10):
	if (scor1_segm_depth[0][ii]/numbers1_segm_depth[0][ii])**0.5 < minds:
		minds = (scor1_segm_depth[0][ii]/numbers1_segm_depth[0][ii])**0.5
print(minds)
print('day:')
minds = 1000000000
for ii in range(10):
	if (scor1_segm_depth[1][ii]/numbers1_segm_depth[1][ii])**0.5 < minds:
		minds = (scor1_segm_depth[1][ii]/numbers1_segm_depth[1][ii])**0.5
print(minds)

print('dusk:')
minds = 1000000000
for ii in range(10):
	if (scor1_segm_depth[2][ii]/numbers1_segm_depth[2][ii])**0.5 < minds:
		minds = (scor1_segm_depth[2][ii]/numbers1_segm_depth[2][ii])**0.5
print(minds)
print('night:')
minds = 1000000000
for ii in range(10):
	if (scor1_segm_depth[3][ii]/numbers1_segm_depth[3][ii])**0.5 < minds:
		minds = (scor1_segm_depth[3][ii]/numbers1_segm_depth[3][ii])**0.5
print(minds)

print('scor depth predicted segm')
print('avg:')
minds = 1000000000
for ii in range(10):
	if (scor1_segm_depth_predicted[0][ii]/numbers1_segm_depth_predicted[0][ii])**0.5 < minds:
		minds = (scor1_segm_depth_predicted[0][ii]/numbers1_segm_depth_predicted[0][ii])**0.5
print(minds)
print('day:')
minds = 1000000000
for ii in range(10):
	if (scor1_segm_depth_predicted[1][ii]/numbers1_segm_depth_predicted[1][ii])**0.5 < minds:
		minds = (scor1_segm_depth_predicted[1][ii]/numbers1_segm_depth_predicted[1][ii])**0.5
print(minds)

print('dusk:')
minds = 1000000000
for ii in range(10):
	if (scor1_segm_depth_predicted[2][ii]/numbers1_segm_depth_predicted[2][ii])**0.5 < minds:
		minds = (scor1_segm_depth_predicted[2][ii]/numbers1_segm_depth_predicted[2][ii])**0.5
print(minds)
print('night:')
minds = 1000000000
for ii in range(10):
	if (scor1_segm_depth_predicted[3][ii]/numbers1_segm_depth_predicted[3][ii])**0.5 < minds:
		minds = (scor1_segm_depth_predicted[3][ii]/numbers1_segm_depth_predicted[3][ii])**0.5
print(minds)

print('scor classes segm')
for ind in range(len(classes)-1):
	print (classes[ind], classes[ind+1])
	minds = 1000000000
	iimin = -1
	for ii in range(10):
		if numbers1_segm_classes[ind][ii]> 0:
			if (scor1_segm_classes[ind][ii]/numbers1_segm_classes[ind][ii])**0.5 < minds:
				minds = (scor1_segm_classes[ind][ii]/numbers1_segm_classes[ind][ii])**0.5
				iimin = ii
	if iimin > -1:
		print(numbers1_segm_classes[ind][iimin])
		print(minds)

print('scor fara segm')
print('avg:')
print((scor2[0]/numbers2[0])**	0.5)
print('day:')
print((scor2[1]/numbers2[1])**	0.5)
print('dusk:')
print((scor2[2]/numbers2[2])**	0.5)
print('night:')
print((scor2[3]/numbers2[3])**	0.5)



print('scor depth fara segm')
print('avg:')
print((scor2_depth[0]/numbers2_depth[0])**0.5)
print('day:')
print((scor2_depth[1]/numbers2_depth[1])**0.5)
print('dusk:')
print((scor2_depth[2]/numbers2_depth[2])**0.5)
print('night:')
print((scor2_depth[3]/numbers2_depth[3])**0.5)

print('scor depth predicted fara segm')
print('avg:')
print((scor2_depth_predicted[0]/numbers2_depth_predicted[0])**0.5)
print('day:')
print((scor2_depth_predicted[1]/numbers2_depth_predicted[1])**0.5)
print('dusk:')
print((scor2_depth_predicted[2]/numbers2_depth_predicted[2])**0.5)
print('night:')
print((scor2_depth_predicted[3]/numbers2_depth_predicted[3])**0.5)


print('scor segm')
print('avg:')
minds = 1000000000
for ii in range(10):
	if (scor2_segm[0][ii]/numbers2_segm[0][ii])**0.5 < minds:
		minds = (scor2_segm[0][ii]/numbers2_segm[0][ii])**0.5
print(minds)
print('day:')
minds = 1000000000
for ii in range(10):
	if (scor2_segm[1][ii]/numbers2_segm[1][ii])**0.5 < minds:
		minds = (scor2_segm[1][ii]/numbers2_segm[1][ii])**0.5
print(minds)

print('dusk:')
minds = 1000000000
for ii in range(10):
	if (scor2_segm[2][ii]/numbers2_segm[2][ii])**0.5 < minds:
		minds = (scor2_segm[2][ii]/numbers2_segm[2][ii])**0.5
print(minds)
print('night:')
minds = 1000000000
for ii in range(10):
	if (scor2_segm[3][ii]/numbers2_segm[3][ii])**0.5 < minds:
		minds = (scor2_segm[3][ii]/numbers2_segm[3][ii])**0.5
print(minds)


print('scor depth segm')
print('avg:')
minds = 1000000000
for ii in range(10):
	if (scor2_segm_depth[0][ii]/numbers2_segm_depth[0][ii])**0.5 < minds:
		minds = (scor2_segm_depth[0][ii]/numbers2_segm_depth[0][ii])**0.5
print(minds)
print('day:')
minds = 1000000000
for ii in range(10):
	if (scor2_segm_depth[1][ii]/numbers2_segm_depth[1][ii])**0.5 < minds:
		minds = (scor2_segm_depth[1][ii]/numbers2_segm_depth[1][ii])**0.5
print(minds)

print('dusk:')
minds = 1000000000
for ii in range(10):
	if (scor2_segm_depth[2][ii]/numbers2_segm_depth[2][ii])**0.5 < minds:
		minds = (scor2_segm_depth[2][ii]/numbers2_segm_depth[2][ii])**0.5
print(minds)
print('night:')
minds = 1000000000
for ii in range(10):
	if (scor2_segm_depth[3][ii]/numbers2_segm_depth[3][ii])**0.5 < minds:
		minds = (scor2_segm_depth[3][ii]/numbers2_segm_depth[3][ii])**0.5
print(minds)

print('scor depth predicted segm')
print('avg:')
minds = 1000000000
for ii in range(10):
	if (scor2_segm_depth_predicted[0][ii]/numbers2_segm_depth_predicted[0][ii])**0.5 < minds:
		minds = (scor2_segm_depth_predicted[0][ii]/numbers2_segm_depth_predicted[0][ii])**0.5
print(minds)
print('day:')
minds = 1000000000
for ii in range(10):
	if (scor2_segm_depth_predicted[1][ii]/numbers2_segm_depth_predicted[1][ii])**0.5 < minds:
		minds = (scor2_segm_depth_predicted[1][ii]/numbers2_segm_depth_predicted[1][ii])**0.5
print(minds)

print('dusk:')
minds = 1000000000
for ii in range(10):
	if (scor2_segm_depth_predicted[2][ii]/numbers2_segm_depth_predicted[2][ii])**0.5 < minds:
		minds = (scor2_segm_depth_predicted[2][ii]/numbers2_segm_depth_predicted[2][ii])**0.5
print(minds)
print('night:')
minds = 1000000000
for ii in range(10):
	if (scor2_segm_depth_predicted[3][ii]/numbers2_segm_depth_predicted[3][ii])**0.5 < minds:
		minds = (scor2_segm_depth_predicted[3][ii]/numbers2_segm_depth_predicted[3][ii])**0.5
print(minds)




print('scor classes segm')
for ind in range(len(classes)-1):
	print (classes[ind], classes[ind+1])
	minds = 1000000000
	iimin = -1
	for ii in range(10):
		if numbers2_segm_classes[ind][ii]> 0:
			if (scor2_segm_classes[ind][ii]/numbers2_segm_classes[ind][ii])**0.5 < minds:
				minds = (scor2_segm_classes[ind][ii]/numbers2_segm_classes[ind][ii])**0.5
				iimin = ii
	if iimin > -1:
		print(numbers2_segm_classes[ind][iimin])
		print(minds)
	
		


