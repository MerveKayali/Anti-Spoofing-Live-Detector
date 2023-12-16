import os
import random
import shutil
from itertools import islice

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/all"
classes = ["fake", "real"]

splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}

try:
    shutil.rmtree(outputFolderPath)
    print("Removed Directory")
except OSError as e:
    os.mkdir(outputFolderPath)

# ---------- Dosya Oluşturma ------------------
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

# ----------------- isimleri alma -------------------
listNames = os.listdir(inputFolderPath)
uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split(".")[0])
uniqueNames = list(set(uniqueNames))

# ------------Shuffle----------------
random.shuffle(uniqueNames)

# -------- Her dosyadaki fotoğraf sayısını bulma-------------
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

# --------------Dataları dosyalara ayırıken eksik kalan fotoğrafları ekleme-------
if lenData != lenTrain + lenVal + lenTest:
    remaining = lenData - (lenTrain + lenVal + lenTest)
    lenTrain += remaining

# -------- listeleri ayırma ---------------
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images:{lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')

# --------Dosyalara kopyalama---------------
sequence = ["train", "val", "test"]
for i, out in enumerate(Output):
    for filename in out:
        shutil.copy(f"{inputFolderPath}/{filename}.jpg", f"{outputFolderPath}/{sequence[i]}/images/{filename}.jpg")
        shutil.copy(f"{inputFolderPath}/{filename}.txt", f"{outputFolderPath}/{sequence[i]}/labels/{filename}.txt")

print("Split process completed...")

# ---------------Data.yaml dosyası oluşturma-------------------
dataYaml = f'path: ../Data\n\
train:  ../train/images\n\
val:  ../val/images\n\
test:  ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'

f = open(f'{outputFolderPath}/data.yaml', 'a')
f.write(dataYaml)
f.close()

print("Data.yaml file created...")
