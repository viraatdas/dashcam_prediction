import csv
import pickle

pkl_file = open('data.pkl', 'rb')

data1 = pickle.load(pkl_file)

with open("data/img_csv/train_info.csv", 'w') as file:
    wr = csv.writer(file, delimiter=',')
    for row in data1:
        wr.writerow(row.split(","))
