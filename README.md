# Summary
The following project was developed as part of my bachelor's thesis. The project implements, trains and evaluates the four best performing Convolutional Neural Networks architectures for the databases RAF-DB, AffectNet, FERG, GoogleCC0 (Google Images). For more information on each database and architecture, see paperswithcode: https://paperswithcode.com/task/facial-expression-recognition

# Databases
For an external evaluation of the databases, two database were not used for productive training, but only for evaluation (JAFFE and FER2013). All databases were legitimately requested and collected from the following sites: 
JAFFE: https://zenodo.org/record/3451524#.YmNuNejP0uU 
AffectNet: http://mohammadmahoor.com/affectnet/ --> https://www.kaggle.com/datasets/mouadriali/affectnetsample ?
FERG_DB_256: http://grail.cs.washington.edu/projects/deepexpr/ferg-2d-db.html
RAF-DB: http://www.whdeng.cn/raf/model1.html
FER2013: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Google: Just execute training/Google/collectGoogle.py

# Client Server Application
Furthermore, a client/server web application was developed and published for real-time use with Flask. Here you can test the models yourself on video or webcam and view information about the respective architectures. The page can be reached under the following link:

To simulate the web application or to test the models individually without having to perform the training yourself, you need to download the checkpoints of the models. The checkpoints of the respective models allow Pytorch to initialize the models with the pre-trained weights without training. To do this, please download the folder provided with this link, unzip it and place the " Checkpoints" folder inside the "training" folder:



