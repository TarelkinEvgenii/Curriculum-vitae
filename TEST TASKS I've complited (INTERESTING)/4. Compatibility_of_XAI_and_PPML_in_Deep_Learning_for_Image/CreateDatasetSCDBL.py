import os
import shutil
import pandas as pd

for file_name  in  os.listdir("./labels_only/") :
    print(file_name)
    f_train=open("./labels_only/"+file_name)
    #csvfile = pd.read_csv('path_to_file',)
    csvfile=pd.read_csv("./labels_only/"+file_name, delimiter='|', names=['image', 'label'], encoding='utf-8')
    print("\n\n\n\n")
    if  not  os.path.exists("./datasets/") :
        os.mkdir("./datasets/")
    if  not  os.path.exists("./datasets/"+file_name.split(".")[0]):
        os.mkdir("./datasets/"+file_name.split(".")[0])

    for i in range(len(csvfile.iloc[:,0])): #I know it's not a pythonic way

        print(csvfile.iloc[i,0])

        if not os.path.exists("./datasets/" + file_name.split(".")[0] + "/" + str(csvfile.iloc[i,1])):
            os.mkdir("./datasets/" + file_name.split(".")[0] + "/" + str(csvfile.iloc[i,1]))
        shutil.copy2( "./"+csvfile.iloc[i,0],"./datasets/" + file_name.split(".")[0] + "/"+str(csvfile.iloc[i,1]) + "/" + csvfile.iloc[i,0].split("/")[-1])