import os, csv

f=open("data/test_info.csv",'w')
w=csv.writer(f)
ls_file = []
for path, dirs, files in os.walk("data/test_image"):
    for filename in files:
        ls_file.append(filename)

ls_file = sorted(ls_file)

for fil in ls_file:
    w.writerow([
