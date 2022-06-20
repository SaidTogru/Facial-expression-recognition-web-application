# Summary

The following project was developed as part of my bachelor's thesis: www.google.com </br></br>The project implements, trains and evaluates the four best performing Convolutional Neural Networks architectures for the databases RAF-DB, AffectNet, FERG, GoogleCC0 (Google Images). For more information of different architectures and their accuracy ranking in the Facial Expression Recognition domain, see paperswithcode: https://paperswithcode.com/task/facial-expression-recognition

# Papers and Architectures

The following paper implementations were used as part of this research:
</br></br>
https://arxiv.org/pdf/1902.01019v1.pdf </br>

https://arxiv.org/pdf/2109.07270v4.pdf</br>

https://openaccess.thecvf.com/content/WACV2021/papers/Farzaneh_Facial_Expression_Recognition_in_the_Wild_via_Deep_Attentive_Center_WACV_2021_paper.pdf</br> </br>
The fourth model "BasicNet" was developed by myself and was also trained on the self-developed database GoogleCC0. ![BasicNet](https://user-images.githubusercontent.com/65668541/174686382-69b7001f-5d90-4f12-aa4b-a7a4153e20b0.png)

The construction of this database was done by developing a web scraper that searches and collects images with the seven emotions (angry, disgust, fear, happy, neutral, sad, surprise) in Google royalty free on 109 languages: https://github.com/SaidTogru/Facial-expression-recognition-web-application/blob/main/useful/GoogleAutomation/collectGoogle.py </br>Some lines of code may need to be adapted due to the rapidly changing web components.

# Databases

For an external evaluation of the databases, two database were not used for productive training, but only for evaluation (JAFFE and FER2013). All databases were legitimately requested and collected from the following organisations: </br></br>
JAFFE: https://zenodo.org/record/3451524#.YmNuNejP0uU </br></br>
AffectNet: http://mohammadmahoor.com/affectnet/ </br></br>
FERG_DB_256: http://grail.cs.washington.edu/projects/deepexpr/ferg-2d-db.html </br></br>
RAF-DB: http://www.whdeng.cn/raf/model1.html </br></br>
FER2013: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data </br></br>

# Client Server Application

Furthermore, a client/server web application was developed and published for real-time usage with Flask. Here you can test the models yourself on video or webcam and view information about the respective architectures. The page can be reached under the following link: </br>

To simulate the web application or test the models individually on your local computer without having to do the training yourself, you need to download the checkpoints of the models. The checkpoints of the respective models allow Pytorch to initialize the models with the pre-trained weights without training. To do this, please download the folder provided with this link, unzip it and place the " Checkpoints" folder inside the "training" folder: https://drive.google.com/file/d/1oE8qyF7ntrs-ZDIPV5KJc2Ur9SaL7lyy/view?usp=sharing
