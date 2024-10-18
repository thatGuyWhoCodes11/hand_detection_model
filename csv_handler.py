import csv
import numpy as np
import os
#a function to create and prepare layout of the csv file
def make_csv(name):
    num_coords = 21
    landmarks = ["class"]
    for i in range(1,num_coords+1):
        #,"z{}".format(i) removed to reduce the bias coming from depth
        landmarks += ["x{}".format(i),"y{}".format(i)]
    with open(name+'.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)
#function to write all hand landmarks into said csv file
def write_to_csv(csv_path,class_name,model_results):
    if not csv_path+".csv" in os.listdir("./"):
        make_csv(csv_path)
    hand_coords=model_results
    normalized_coords = np.array([[hand_coord.x,hand_coord.y] for hand_coord in hand_coords])
    #hand_coord.z removed to reduce the bias coming from depth
    hand_coords_row = list(normalized_coords.flatten())
    hand_coords_row.insert(0,class_name)
    with open(csv_path+'.csv', mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(hand_coords_row)
#reduce the bias of the data by having all landmarks on one point of origin