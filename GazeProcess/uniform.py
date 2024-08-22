import csv
import os

imagepath =r'D:\csx\ScanPath\AOI_600v2\image_test'
datapath = r'D:\csx\ScanPath\AOI_600v2\1.5_100'
savepath = r'D:\csx\ScanPath\AOI_600\uniform'


# imagepath =r'D:\csx\ScanPath\ScanDMM-Sitzmann\Sitzmann\rotation_imgs\imgs'
# datapath = r'D:\csx\ScanPath\sitzmann\data\1.5_100'
# savepath = r'D:\csx\ScanPath\sitzmann\uniform'

for imagename in os.listdir(imagepath):
    imagename = imagename[:-4]
    for txt_name in os.listdir(os.path.join(datapath,imagename)):
        f = open(os.path.join(datapath,imagename,txt_name),'r')
        f = f.readlines()
        reader = f[29:]
        data = []
        for i in range(0,len(reader),70):
            content = reader[i].replace('\n','').split(',')
            data.append(content[0]+';'+content[1]+';'+content[2])
        with open(os.path.join(savepath,imagename+'.csv'), 'a', newline='') as csvfile1:
            writer = csv.writer(csvfile1)
            writer.writerow(data)