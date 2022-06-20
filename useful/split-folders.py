import splitfolders
import os
import shutil

# jaffe
label = {'an': 'angry', 'di': 'disgust', 'fe': 'fear',
         'ha': 'happy', 'ne': 'neutral', 'sa': 'sad', 'su': 'suprise'}
path = "jaffe\\jaffedbase\\"
for e in label.values():
    if not os.path.exists('jaffe\\Input\\'+e):
        os.makedirs('jaffe\\Input\\'+e)
for file in os.listdir(path):
    if file.endswith(".tiff"):
        folder = label[file.split(".")[1][:2].lower()]
        shutil.copy2(path+file, 'jaffe\\Input\\'+folder+"\\")


# FERG_DB_256
label = {'anger': 'angry', 'disgust': 'disgust', 'fear': 'fear',
         'joy': 'happy', 'sadness': 'sad', 'surprise': 'surprise', 'neutral': 'neutral'}
folder_list = [x[0]
               for x in os.walk("FERG_DB_256") if len(x[0].split('\\')) > 2]
for e in label.values():
    if not os.path.exists('FERG_DB_256\\Input\\'+e):
        os.makedirs('FERG_DB_256\\Input\\'+e)
for folder in folder_list:
    for file in os.listdir(folder):
        emotion = label[folder.rsplit("_")[-1]]
        shutil.copy2(
            folder+"\\"+file, 'FERG_DB_256\\Input\\'+emotion+'\\')
        quit()
splitfolders.ratio("FERG_DB_256\\Input", output="FERG_DB_256\\Output",
                   seed=1337, ratio=(.8, .2), group_prefix=None, move=False)


# RAF-DB
label = {"1": "surprise", "2": "fear", "3": "disgust",
         "4": "happy", "5": "sad", "6": "angry", "7": "neutral"}
file_annotation = {}
path = "RAF-DB\\Image\\aligned\\"
with open("RAF-DB\\EmoLabel\\list_patition_label.txt", "r") as myfile:
    for line in myfile:
        (key, val) = line.split()
        file_annotation[key] = val
for e in label.values():
    if not os.path.exists('RAF-DB\\Input\\'+e):
        os.makedirs('RAF-DB\\Input\\'+e)
for file in os.listdir(path):
    folder = label[file_annotation[file.replace("_aligned", "")]]
    shutil.copy2(
        path+file, 'RAF-DB\\Input\\'+folder+"\\")
splitfolders.ratio("RAF-DB\\Input", output="RAF-DB\\Output",
                   seed=1337, ratio=(.8, .2), group_prefix=None, move=False)

# AffectNet
"""Just manually rename the folders to the corresponding expression and the delete contempt folder"""

# Google
"""
Collected Date with Selenium on Google Images CC0 and validated them with easy CNN before saving.
Searched on all available languages
"""
