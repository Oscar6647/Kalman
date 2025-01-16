from roboflow import Roboflow

rf = 
project = rf.workspace("lab1-1djxq").project("lab_2-sm2mr")
version = project.version(1)
dataset = version.download("yolov8")
                
