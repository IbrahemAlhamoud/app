from tkinter import *
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, make_response,render_template
from flask_sqlalchemy import SQLAlchemy
import base64 
import tensorflow as tf
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt
import shutil as sh
from flask import Flask, jsonify, request, make_response,render_template
from flask_sqlalchemy import SQLAlchemy
import base64
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.preprocessing import image
from tensorflow import keras
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil as sh
from flask import Flask, jsonify, request, make_response,render_template
from flask_sqlalchemy import SQLAlchemy
import base64 

def iw():
  called = True
  if called:
        count_file = open("glioma_tumor.txt", "r") # open file in read mode
        count = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("glioma_tumor.txt", "w") # open file again but in write mode
        count = int(count) + 1 # increase the count value by add 1
        count_file.write(str(count)) # write count to file
        count_file.close() # close file
        return count
def i1w():        
 called1 = True
 if called1:
        count_file = open("meningioma_tumor.txt", "r") # open file in read mode
        count1 = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("meningioma_tumor.txt", "w") # open file again but in write mode
        count1 = int(count1) + 1 # increase the count value by add 1
        count_file.write(str(count1)) # write count to file
        count_file.close() # close file
        return count1

def i2w():        
 called1 = True
 if called1:
        count_file = open("no_tumor.txt", "r") # open file in read mode
        count2 = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("no_tumor.txt", "w") # open file again but in write mode
        count2 = int(count2) + 1 # increase the count value by add 1
        count_file.write(str(count2)) # write count to file
        count_file.close() # close file
        return count2   
    
def i3w():        
 called1 = True
 if called1:
        count_file = open("pituitary_tumor.txt", "r") # open file in read mode
        count3 = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("pituitary_tumor.txt", "w") # open file again but in write mode
        count3 = int(count3) + 1 # increase the count value by add 1
        count_file.write(str(count3)) # write count to file
        count_file.close() # close file
        return count3  
def values6(value1):
  if value1 == [0] : 
    
    i0=iw()  
    sh.move('deer_decode2.jpg' , 'C:/Users/User/Desktop/test brain tumor/1/glioma_tumor_%d.jpg'%i0) 

  if value1 == [1]: 
    i11=i2w() 
    sh.move('deer_decode2.jpg' , 'C:/Users/User/Desktop/test brain tumor/2/no_tumor_%d.jpg'%i11)
  
  if value1 == [2]: 
   i22=i1w() 
   sh.move('deer_decode2.jpg' , 'C:/Users/User/Desktop/test brain tumor/3/meningioma_tumorr_%d.jpg'%i22)

  if value1 == [3]: 
    i33=i3w()
    sh.move('deer_decode2.jpg' , 'C:/Users/User/Desktop/test brain tumor/4/pituitary_tumor_%d.jpg'%i33)    

def ir():
  called = True
  if called:
        count_file = open("benign.txt", "r") # open file in read mode
        count = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("benign.txt", "w") # open file again but in write mode
        count = int(count) + 1 # increase the count value by add 1
        count_file.write(str(count)) # write count to file
        count_file.close() # close file
        return count
def i1r():        
 called1 = True
 if called1:
        count_file = open("malignant.txt", "r") # open file in read mode
        count1 = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("malignant.txt", "w") # open file again but in write mode
        count1 = int(count1) + 1 # increase the count value by add 1
        count_file.write(str(count1)) # write count to file
        count_file.close() # close file
        return count1

def values5(value1):
  if value1 == [0] : 
    
    i0=ir()  
    sh.move('deer_decode2.jpg' , 'C:/Users/User/Desktop/test_skin/1/benign_%d.jpg'%i0) 

  if value1 == [1]: 
    i11=i1r() 
    sh.move('deer_decode2.jpg' , 'C:/Users/User/Desktop/test_skin/2/malignant_%d.jpg'%i11)    

def ie():
  called = True
  if called:
        count_file = open("MildDemented.txt", "r") # open file in read mode
        count = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("MildDemented.txt", "w") # open file again but in write mode
        count = int(count) + 1 # increase the count value by add 1
        count_file.write(str(count)) # write count to file
        count_file.close() # close file
        return count
def i1e():        
 called1 = True
 if called1:
        count_file = open("ModerateDemented.txt", "r") # open file in read mode
        count1 = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("ModerateDemented.txt", "w") # open file again but in write mode
        count1 = int(count1) + 1 # increase the count value by add 1
        count_file.write(str(count1)) # write count to file
        count_file.close() # close file
        return count1

def i2e():        
 called1 = True
 if called1:
        count_file = open("NonDemented.txt", "r") # open file in read mode
        count2 = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("NonDemented.txt", "w") # open file again but in write mode
        count2 = int(count2) + 1 # increase the count value by add 1
        count_file.write(str(count2)) # write count to file
        count_file.close() # close file
        return count2   
    
def i3e():        
 called1 = True
 if called1:
        count_file = open("VeryMildDemented.txt", "r") # open file in read mode
        count3 = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("VeryMildDemented.txt", "w") # open file again but in write mode
        count3 = int(count3) + 1 # increase the count value by add 1
        count_file.write(str(count3)) # write count to file
        count_file.close() # close file
        return count3  
    
    
def values4(value1):
  if value1 == [0] : 
    
    i0=ie()  
    sh.move('deer_decode2.jpg' , 'C:/Users/User/Desktop/test alzheimers/1/MildDemented_%d.jpg'%i0) 

  if value1 == [1]: 
    i11=i1e() 
    sh.move('deer_decode2.jpg' , 'C:/Users/User/Desktop/test alzheimers/2/ModerateDemented_%d.jpg'%i11)
  
  if value1 == [2]: 
   i22=i2e() 
   sh.move('deer_decode2.jpg' , 'C:/Users/User/Desktop/test alzheimers/3/NonDemented_%d.jpg'%i22)

  if value1 == [3]: 
    i33=i3e()
    sh.move('deer_decode2.jpg' , 'C:/Users/User/Desktop/test alzheimers/4/VeryMildDemented_%d.jpg'%i33)    

def ic():
  called = True
  if called:
        count_file = open("healthy 2.txt", "r") # open file in read mode
        count = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("healthy 2.txt", "w") # open file again but in write mode
        count = int(count) + 1 # increase the count value by add 1
        count_file.write(str(count)) # write count to file
        count_file.close() # close file
        return count
def i1c():        
 called1 = True
 if called1:
        count_file = open("Breast_Cancer.txt", "r") # open file in read mode
        count1 = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("Breast_Cancer.txt", "w") # open file again but in write mode
        count1 = int(count1) + 1 # increase the count value by add 1
        count_file.write(str(count1)) # write count to file
        count_file.close() # close file
        return count1

def values3(value1):
  if value1 == [0] : 
    
    i0=ic()  
    sh.move('deer_decode2.png' , 'C:/Users/User/Desktop/test Breast_Cancer/0/healthy_%d.png'%i0) 

  if value1 == [1]: 
    i11=i1c() 
    sh.move('deer_decode2.png' , 'C:/Users/User/Desktop/test Breast_Cancer/1/Breast_Cancer_%d.png'%i11)

def i():
  called = True
  if called:
        count_file = open("count.txt", "r") # open file in read mode
        count = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("count.txt", "w") # open file again but in write mode
        count = int(count) + 1 # increase the count value by add 1
        count_file.write(str(count)) # write count to file
        count_file.close() # close file
        return count
def i1():        
 called1 = True
 if called1:
        count_file = open("count1.txt", "r") # open file in read mode
        count1 = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("count1.txt", "w") # open file again but in write mode
        count1 = int(count1) + 1 # increase the count value by add 1
        count_file.write(str(count1)) # write count to file
        count_file.close() # close file
        return count1
def i2():        
  called2 = True
  if called2:
        count_file = open("count2.txt", "r") # open file in read mode
        count2 = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("count2.txt", "w") # open file again but in write mode
        count2 = int(count2) + 1 # increase the count value by add 1
        count_file.write(str(count2)) # write count to file
        count_file.close() # close file
        return count2
def i3():        
 called3 = True
 if called3:
        count_file = open("count3.txt", "r") # open file in read mode
        count3 = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("count3.txt", "w") # open file again but in write mode
        count3 = int(count3) + 1 # increase the count value by add 1
        count_file.write(str(count3)) # write count to file
        count_file.close() # close file        
        return count3

def ib():
  called = True
  if called:
        count_file = open("healthy1.txt", "r") # open file in read mode
        count = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("healthy1.txt", "w") # open file again but in write mode
        count = int(count) + 1 # increase the count value by add 1
        count_file.write(str(count)) # write count to file
        count_file.close() # close file
        return count
def values2(value1):
  if value1 == 0 : 
    
    i0=i()  
    sh.move('deer_decode1.png' , 'C:/Users/User/Desktop/test/1/covid_%d.png'%i0) 

  if value1 == 1: 
    i11=i1() 
    sh.move('deer_decode1.png' , 'C:/Users/User/Desktop/test/2/Lung_%d'%i11)

    
  if value1 == 2: 
    i22=i2() 
    sh.move('deer_decode1.png' , 'C:/Users/User/Desktop/test/3/NORMAL_%d.png'%i22)
   
  if value1 == 3 :  
    i33=i3()
    sh.move('deer_decode1.png' , 'C:/Users/User/Desktop/test/4/Viral_%d.png'%i33)
    
def i1b():        
 called1 = True
 if called1:
        count_file = open("parkinson1.txt", "r") # open file in read mode
        count1 = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("parkinson1.txt", "w") # open file again but in write mode
        count1 = int(count1) + 1 # increase the count value by add 1
        count_file.write(str(count1)) # write count to file
        count_file.close() # close file
        return count1
    
def values1(value1):
  if value1 == [0] : 
    
    i0=i()  
    sh.move('deer_decode2.png' , 'C:/Users/User/Desktop/parkinson2/0/healthy_%d.png'%i0) 

  if value1 == [1]: 
    i11=i1() 
    sh.move('deer_decode2.png' , 'C:/Users/User/Desktop/parkinson2/1/parkinson_%d.png'%i11)    

def ia():
  called =True
  if called:
        count_file = open("healthy.txt", "r") # open file in read mode
        count = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("healthy.txt", "w") # open file again but in write mode
        count = int(count) + 1 # increase the count value by add 1
        count_file.write(str(count)) # write count to file
        count_file.close() # close file
        return count
def i1a():        
 called1 =True
 if called1:
        count_file = open("parkinson.txt", "r") # open file in read mode
        count1 = count_file.read() # read data 
        count_file.close() # close file

        count_file = open("parkinson.txt", "w") # open file again but in write mode
        count1 = int(count1) + 1 # increase the count value by add 1
        count_file.write(str(count1)) # write count to file
        count_file.close() # close file
        return count1
    
def image_64_encode(image_64_encode1):
   encode_file = open("encode.txt", "w") # open file again but in write mode
   encode_file.write(str(image_64_encode1)) # write count to file
   encode_file.close() # close file
   
   read_file = open("encode.txt", "r") # open file in read mode
   encode1 = read_file.read() # read data 
   read_file.close() # close file

def values(value1):
  if value1 == [0] : 
    
    i0=ia()  
    sh.move('deer_decode2.png' , 'C:/Users/User/Desktop/parkinson1/0/healthy_%d.png'%i0) 

  if value1 == [1]: 
    i11=i1a() 
    sh.move('deer_decode2.png' , 'C:/Users/User/Desktop/parkinson1/1/parkinson_%d.png'%i11)
   



def DecisionTree(p11):
    
    l11=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
    'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
    'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
    'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
    'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
    'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze']
    f=dict()
    i=0
    for a in l11 :
      f[a]=i
      i=i+1
    disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
    'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
    ' Migraine','Cervical spondylosis',
    'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
    'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
    'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
    'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
    'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
    'Impetigo']
    l2=list()
    for x in range(0,len(l11)):
      l2.append(0)
    disease1={0:'Fungal infection',1:'Allergy',2:'GERD',3:'Chronic cholestasis',4:'Drug Reaction',
    5:'Peptic ulcer diseae',6:'AIDS',7:'Diabetes',8:'Gastroenteritis',9:'Bronchial Asthma',10:'Hypertension',
    11:' Migraine',12:'Cervical spondylosis',
    13:'Paralysis (brain hemorrhage)',14:'Jaundice',15:'Malaria',16:'Chicken pox',17:'Dengue',18:'Typhoid',19:'hepatitis A',
    20:'Hepatitis B',21:'Hepatitis C',22:'Hepatitis D',23:'Hepatitis E',24:'Alcoholic hepatitis',25:'Tuberculosis',
    26:'Common Cold',27:'Pneumonia',28:'Dimorphic hemmorhoids(piles)',
    29:'Heartattack',30:'Varicoseveins',31:'Hypothyroidism',32:'Hyperthyroidism',33:'Hypoglycemia',34:'Osteoarthristis',
    35:'Arthritis',36:'(vertigo) Paroymsal  Positional Vertigo',37:'Acne',38:'Urinary tract infection',39:'Psoriasis',
    40:'Impetigo'}

    tr=pd.read_csv("C:/Users/User/Desktop/مشروع التخرج/Disease-prediction/Disease-prediction/Testing.csv")
    tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
    'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
    'Migraine':11,'Cervical spondylosis':12,
    'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
    'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
    'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
    'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
    '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
    'Impetigo':40}},inplace=True)
  
    X_test= tr[l11]
    y_test = tr[["prognosis"]]
    np.ravel(y_test)
    
    from sklearn import tree

    filename ="disease_model.sav"
    loaded_model = joblib.load(filename)


    from sklearn.metrics import accuracy_score
    y_pred=loaded_model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    #print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------
    psymptoms = p11
    
    o=len(psymptoms)
    if(o==0):
      print("Not Found")
      
    else : 
      for k in f:
        # print (k,)
        for z in psymptoms:
            
            if(z==f[k]):
                l2[f[k]]=1

      inputtest = [l2]
      print(l2)
      predict = loaded_model.predict(inputtest)
      predicted=predict[0]

      h='no'
      for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


      if (h=='yes'):
        print(a)  
        print(disease[a])
        return(disease[a])
        
      else:
        print("Not Found")

def RandomForest(p11):
    
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    from flask import Flask, jsonify, request, make_response,render_template
    from flask_sqlalchemy import SQLAlchemy
    import base64 


    df = pd.read_csv('C:/Users/User/Desktop/مشروع التخرج/Breast_cancer_data.csv')
    df.head()

    x = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    from sklearn import model_selection
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    # random forest model creation
    rfc = RandomForestClassifier()
    rfc.fit(x_train,y_train)
    # predictions
    rfc_predict = rfc.predict(x_test)
    print("Accuracy:",accuracy_score(y_test, rfc_predict))
    patient_sample = np.array([p11])
    prediction = rfc.predict(patient_sample)
    
    if prediction == 0:
        print("You are not expected to be Breast Cancer")
        
    elif prediction == 1:
        print("You are expected to be Breast Cancer")         
    
    return (prediction)

def Gradient_Boosting(p11):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy as sp
    import warnings
    import os 
    warnings.filterwarnings("ignore")
    import datetime
    from flask import Flask, jsonify, request, make_response,render_template
    from flask_sqlalchemy import SQLAlchemy
    import base64 

    data=pd.read_csv('C:/Users/User/Desktop/مشروع التخرج/data.csv')
    data.head()      #displaying the head of dataset
 
    data.drop('Unnamed: 32', axis = 1, inplace = True)
    data.corr()
    #a=[14.68,20.13,94.74,684.5,0.09867,0.072,0.07395,0.05259,0.1586,0.05922,0.4727,1.24,3.195,45.4,0.005718,0.01162,0.01998,0.01109,0.0141,0.002085,19.07,30.88,123.4,1138,0.1464,0.1871,0.2914,0.1609,0.3029,0.08216]
    # Getting Features

    x = data.drop(columns = ['diagnosis','id'],axis=1)
    print(x)
    # Getting Predicting Value
    y = data['diagnosis']
    #train_test_splitting of the dataset
    from sklearn.model_selection import train_test_split 
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    
    
    from sklearn.ensemble import GradientBoostingClassifier
    gbc=GradientBoostingClassifier()
    gbc.fit(x_train,y_train)

    y_pred=gbc.predict(x_test)
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_squared_error,r2_score
    #print(classification_report(y_test,y_pred))
    #print(confusion_matrix(y_test,y_pred))
    #print("Training Score: ",gbc.score(x_train,y_train)*100)
    #print(gbc.score(x_test,y_test))
    print(accuracy_score(y_test,y_pred)*100)
            #----------------------------------------------------------
    patient_sample = np.array([p11])
    prediction =gbc.predict(patient_sample)
    print(prediction)
    if prediction == ['B']:
        print("You are not expected to be Breast Cancer")
        
    elif prediction == ['M']:
        print("You are expected to be Breast Cancer")
    return (prediction)

def Logistic_Regression(p11):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix 

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from flask import Flask, jsonify, request, make_response,render_template
    from flask_sqlalchemy import SQLAlchemy
    import base64 


    from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score
    #import Dataset
    dataset = pd.read_csv('C:/Users/User/Desktop/مشروع التخرج/kidney_disease.csv')
    #print(dataset.head())


    #a=[55.0,80.0,1.02,0.0,0.0,133.0,17.0,1.2,135.0,4.8,13.2,41.0,6800.0,5.3]
    #print(len(a))


    dataset[['htn','dm','cad','pe','ane']]=dataset[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
    dataset[['rbc','pc']] = dataset[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
    dataset[['pcc','ba']] = dataset[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
    dataset[['appet']] = dataset[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
    dataset['classification']=dataset['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})
    dataset.rename(columns={'classification':'class'},inplace=True)
    # Further cleaning
    dataset['pe'] = dataset['pe'].replace(to_replace='good',value=0) # Not having pedal edema is good
    dataset['appet'] = dataset['appet'].replace(to_replace='no',value=0)
    dataset['cad'] = dataset['cad'].replace(to_replace='\tno',value=0)
    dataset['dm'] = dataset['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})
    dataset.drop('id',axis=1,inplace=True)

    # '?' character remove process in the dataset
    for i in ['rc','wc','pcv']:
      dataset[i] = dataset[i].str.extract('(\d+)').astype(float)
    # Filling missing numeric data in the dataset with mean
    for i in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:
      dataset[i].fillna(dataset[i].mean(),inplace=True)
    dataset.isnull().sum()


    dataset = dataset.dropna(axis=1) 

    #Data preprocessing
    X = dataset.drop(columns=['class'])
    print(X)
    y = dataset['class']
    print(y)
    # Feature Scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)
    #Splitting the dataset in to training and testing set
    X_train , X_test , y_train , y_test   = train_test_split(X,y,test_size = 0.2 , random_state=123) 
    
    
    # Training the Logistic Regression model on the Training set
    lg = LogisticRegression(random_state = 0)
    lg.fit(X_train, y_train)
    #predictin the test result
    y_pred_lg = lg.predict(X_test) 
    #calculate accuracy
    score_lg = accuracy_score(y_pred_lg,y_test)
    print("train score - " + str(lg.score(X_train, y_train)))
    print("test score - " + str(lg.score(X_test, y_test)))
    case = np.array(p11).reshape((1,-1))
    res=lg.predict(case)[0]
    print(res)
    if res == 0.0:
        print("You are not expected to be CKD")
        
    elif res == 1.0:
        print("You are expected to be CKD")    
    return res  

def AdaBoost(p11): 
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split,GridSearchCV
    from sklearn.metrics import accuracy_score,confusion_matrix
    from flask import Flask, jsonify, request, make_response,render_template
    from flask_sqlalchemy import SQLAlchemy
    import base64 

    import warnings
    warnings.filterwarnings('ignore')

    sns.set()

    data = pd.read_csv('C:/Users/User/Desktop/مشروع التخرج/diabetes.csv')  #import dataset
    data.head(10)

    data.shape
 
    data.info()

    data.describe()
    data.isnull().sum()

    X = data.drop('Outcome',axis=1)
    y = data['Outcome'] 
    from sklearn.model_selection import train_test_split        
    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=.30,random_state=3)
    
    from sklearn.ensemble import AdaBoostClassifier
    adb = AdaBoostClassifier(base_estimator = None)
    adb.fit(X_train,Y_train)

    y_pred=adb.predict(X_test)
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,mean_squared_error,r2_score
    (classification_report(Y_test,y_pred))
    #print(confusion_matrix(Y_test,y_pred))
    print("Training Score: ",adb.score(X_train,Y_train)*100)
    
    print(accuracy_score(Y_test,y_pred)*100)
        #----------------------------------------------------------
    patient_sample = np.array([p11])
    prediction =adb.predict(patient_sample)
    print(prediction)
    if prediction == 0:
        print("You are not expected to be diabetic")
        
        
    elif prediction == 1:
        print("You are expected to be diabetic")
    return (prediction) 

def parkinson1(p11):
    
    
    
 image_64_encode(p11) 
 image_64_decode = base64.decodestring(bytes(p11[1:],'utf-8')) #b1
 image_result = open('deer_decode2.png', 'wb') # create a writable image and write the decoding result
 image_result.write(image_64_decode)
 image_result.close() # close file    
  
  
 s1=tf.keras.models.load_model("F:/model_parkinson1.h5") 

  
 #folder_benign_train = 'C:/Users/User/Desktop/مشروع التخرج/مجلد جديد/archive(1)_2/data1'
 #e=os.listdir(folder_benign_train)
 
 c='deer_decode2.png'
 new_img = image.load_img(c, target_size=(100,100))
 plt.imshow(new_img)
 img = image.img_to_array(new_img)
 img=img/255.0
 img=np.array(img)
 img = np.expand_dims(img, axis=0)
 prediction =s1.predict(img)
 result  = np.argmax(prediction,axis=1)
 print(result)
 if result == [0] :
     print('healthy')
 elif result == [1] :
     print('parkinson')
 
 return (result)     


def parkinson2(p11):
 image_64_encode(p11) 
 image_64_decode = base64.decodestring(bytes(p11[1:],'utf-8')) #b1
 image_result = open('deer_decode2.png', 'wb') # create a writable image and write the decoding result
 image_result.write(image_64_decode)
 image_result.close() # close file    
  
  
 s2=tf.keras.models.load_model("F:/model_parkinson2.h5") 

  
 #folder_benign_train = 'C:/Users/User/Desktop/مشروع التخرج/مجلد جديد/archive(1)_2/data1'
 #e=os.listdir(folder_benign_train)
 
 c='deer_decode2.png'
 new_img = image.load_img(c, target_size=(100,100))
 plt.imshow(new_img)
 img = image.img_to_array(new_img)
 img=img/255.0
 img=np.array(img)
 img = np.expand_dims(img, axis=0)
 prediction =s2.predict(img)
 result  = np.argmax(prediction,axis=1)
 print(result)
 if result == [0] :
     print('healthy')
 elif result ==[1] :
     print('parkinson')
 
 return (result)

def covid(p11):
  image_64_encode(p11)
  image_64_decode = base64.decodestring(bytes(p11[1:],'utf-8')) #b1
  image_result = open('deer_decode1.png', 'wb') # create a writable image and write the decoding result
  image_result.write(image_64_decode)
  image_result.close() # close file 
  
  s=tf.keras.models.load_model("F:/covid_2019.hdf5") 
  class_names=["COVID-19","Lung_Opacity","NORMAL","Viral Pneumonia"]


  #folder_benign_train = 'C:/Users/User/Downloads/Compressed/covid_test'
  #e=os.listdir(folder_benign_train)
  from keras.preprocessing import image
  '''
  for r in e :
  image_path = "C:/Users/User/Downloads/Compressed/covid_test/%s"%r

  new_img = image.load_img(image_path, target_size=(224, 224))
  img = image.img_to_array(new_img)
  img=img/255.0
  img=np.array(img)
  img = np.expand_dims(img, axis=0)
  prediction = s.predict(img)
  value = np.argmax(prediction,axis=1)
  print(prediction)
  print(class_names[value[0]])
  plt.imshow(new_img)
  '''
  new_img = image.load_img('deer_decode1.png', target_size=(224, 224))
  img = image.img_to_array(new_img)
  img=img/255.0
  img=np.array(img)
  img = np.expand_dims(img, axis=0)
  prediction = s.predict(img)
  value = np.argmax(prediction,axis=1)
  print(value)
  print(class_names[value[0]])
  plt.imshow(new_img)
  '''
   if os.listdir('C:/Users/User/Desktop/test/1') == [] and value == 0 :
    i1=i()
    
    sh.move('deer_decode.png' , 'C:/Users/User/Desktop/test/1/covid_%d.png'%i1) 
    sh.rmtree('deer_decode.png')
   elif  os.listdir('C:/Users/User/Desktop/test/2') == [] and value == 1 :   
    i2=i1()
    sh.move('deer_decode.png' , 'C:/Users/User/Desktop/test/2/Lung_%d.png'%i2) 
    sh.rmtree('deer_decode.png')
   elif  os.listdir('C:/Users/User/Desktop/test/3') == [] and value == 2 :   
    i3=i2()
    sh.move('deer_decode.png' , 'C:/Users/User/Desktop/test/3/NORMAL_%d.png'%i3) 
    sh.rmtree('deer_decode.png')
   elif  os.listdir('C:/Users/User/Desktop/test/4') == [] and value == 3 :   
    i4=i3()
    sh.move('deer_decode.png' , 'C:/Users/User/Desktop/test/4/Viral_%d.png'%i4) 
    sh.rmtree('deer_decode.png')
    
   '''
  return value

def Breast_Cancer(p11):
 image_64_encode(p11) 
 image_64_decode = base64.decodestring(bytes(p11[1:],'utf-8')) #b1
 image_result = open('deer_decode2.png', 'wb') # create a writable image and write the decoding result
 image_result.write(image_64_decode)
 image_result.close() # close file    
  
  
 s=tf.keras.models.load_model("F:/mcnn (1).h5") 

  
 #folder_benign_train = 'C:/Users/User/Desktop/مشروع التخرج/مجلد جديد/archive(1)_2/data1'
 #e=os.listdir(folder_benign_train)
 
 c='deer_decode2.png'
 new_img = image.load_img(c, target_size=(50,50))
 plt.imshow(new_img)
 img = image.img_to_array(new_img)
 img=img/255.0
 img=np.array(img)
 img = np.expand_dims(img, axis=0)
 prediction =s.predict(img)
 result  = np.argmax(prediction,axis=1)
 print(result)
 if result == [0] :
     print('healthy')
 elif result == [1] :
     print('Breast_Cancer')
 
 return (result)

def alzheimers(p11):
 image_64_encode(p11) 
 image_64_decode = base64.decodestring(bytes(p11[1:],'utf-8')) #b1
 image_result = open('deer_decode2.jpg', 'wb') # create a writable image and write the decoding result
 image_result.write(image_64_decode)
 image_result.close() # close file    
  
  
 s=tf.keras.models.load_model("F:/model_alzheimers.h5") 

  
 #folder_benign_train = 'C:/Users/User/Desktop/مشروع التخرج/مجلد جديد/archive(1)_2/data1'
 #e=os.listdir(folder_benign_train)
 
 c='deer_decode2.jpg'
 new_img = image.load_img(c, target_size=(176, 208))
 plt.imshow(new_img)
 img = image.img_to_array(new_img)
 img=img/255.0
 img=np.array(img)
 img = np.expand_dims(img, axis=0)
 prediction =s.predict(img)
 result  = np.argmax(prediction,axis=1)
 print(result)
 if result == [0]:
     print("MildDemented")
     
 elif result == [1]:
     print("ModerateDemented")
     
 elif result == [2]:
      print("NonDemented")
 elif result == [3]:

      print("VeryMildDemented")
 
 return (result)

def skin(p11):
 image_64_encode(p11) 
 image_64_decode = base64.decodestring(bytes(p11[1:],'utf-8')) #b1
 image_result = open('deer_decode2.jpg', 'wb') # create a writable image and write the decoding result
 image_result.write(image_64_decode)
 image_result.close() # close file    
  
  
 s=tf.keras.models.load_model("F:/model_Skin_cancer_vgg16.h5") 

  
 #folder_benign_train = 'C:/Users/User/Desktop/مشروع التخرج/مجلد جديد/archive(1)_2/data1'
 #e=os.listdir(folder_benign_train)
 
 c='deer_decode2.jpg'
 new_img = image.load_img(c, target_size=(224,224))
 plt.imshow(new_img)
 img = image.img_to_array(new_img)
 img=img/255.0
 img=np.array(img)
 img = np.expand_dims(img, axis=0)
 prediction =s.predict(img)
 result  = np.argmax(prediction,axis=1)
 print(result)
 if result ==[0] :
     print('benign')
 elif result ==[1] :
     print('malignant')
 
 return (result)
  
def brain_tumor(p11):
 image_64_encode(p11) 
 image_64_decode = base64.decodestring(bytes(p11[1:],'utf-8')) #b1
 image_result = open('deer_decode2.jpg', 'wb') # create a writable image and write the decoding result
 image_result.write(image_64_decode)
 image_result.close() # close file    
  
  
 s=tf.keras.models.load_model("F:/model_brain_tumor.h5") 

  
 #folder_benign_train = 'C:/Users/User/Desktop/مشروع التخرج/مجلد جديد/archive(1)_2/data1'
 #e=os.listdir(folder_benign_train)
 
 c='deer_decode2.jpg'
 new_img = image.load_img(c, target_size=(150,150))
 plt.imshow(new_img)
 img = image.img_to_array(new_img)
 img = np.expand_dims(img, axis=0)
 prediction =s.predict(img)
 result  = np.argmax(prediction,axis=1)
 print(result)
 if result == [0]:
     print("glioma_tumor")
     
 elif result == [1]:
     print("no_tumor")
     
 elif result == [2]:
      print("meningioma_tumor")
 elif result == [3]:

      print("pituitary_tumor")
 
 return (result)
#-------------------------

app = Flask(__name__)     
@app.route('/books', methods=['POST'])
def add_get_books():
  b1= request.json['1']
  b2 = request.json['2']
  b3= request.json['3']
  b4= request.json['4']
  b5= request.json['5']
  p1=[b1,b2,b3,b4,b5]
  print(p1)      
  y=str(DecisionTree(p1))
  return jsonify(y)

@app.route('/books1', methods=['POST'])
def add_get_books1():
  b1= float(request.json['1'])
  b2 = float(request.json['2'])
  b3= float(request.json['3'])
  b4= float(request.json['4'])
  b5= float(request.json['5'])
  p1=[b1,b2,b3,b4,b5]
  print(p1)      
  y=RandomForest(p1)
  print("_--------------------------------------------_")
  if y== 0:

        return jsonify("You are (not) expected to Breast Cancer")
        
  elif y == 1:
          
        return jsonify("You are expected to be Breast Cancer")
    
@app.route('/books2', methods=['POST'])
def add_get_books2():
  b1= float(request.json['1'])
  b2 = float(request.json['2'])
  b3= float(request.json['3'])
  b4= float(request.json['4'])
  b5= float(request.json['5'])
  b6= float(request.json['6'])
  b7= float(request.json['7'])
  b8= float(request.json['8'])
  b9= float(request.json['9'])
  b10= float(request.json['10'])
  b11= float(request.json['11'])
  b12= float(request.json['12'])
  b13= float(request.json['13'])
  b14= float(request.json['14'])
  b15= float(request.json['15'])
  b16= float(request.json['16'])
  b17= float(request.json['17'])
  b18= float(request.json['18'])
  b19= float(request.json['19'])
  b20= float(request.json['20'])
  b21= float(request.json['21'])
  b22= float(request.json['22'])
  b23= float(request.json['23'])
  b24= float(request.json['24'])
  b25= float(request.json['25'])
  b26= float(request.json['26'])
  b27= float(request.json['27'])
  b28= float(request.json['28'])
  b29= float(request.json['29'])
  b30= float(request.json['30'])
  p1=[b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30]
  print(p1)      
  
  y=Gradient_Boosting(p1)
  print("_--------------------------------------------_")
  if y== ['B']:

        return jsonify("You are (not) expected to Breast Cancer")
        
  elif y == ['M']:
          
        return jsonify("You are expected to be Breast Cancer")    
    
@app.route('/books3', methods=['POST'])
def add_get_books3():
  b1= float(request.json['1'])
  b2 = float(request.json['2'])
  b3= float(request.json['3'])
  b4= float(request.json['4'])
  b5= float(request.json['5'])
  b6= float(request.json['6'])
  b7= float(request.json['7'])
  b8= float(request.json['8'])
  b9= float(request.json['9'])
  b10= float(request.json['10'])
  b11= float(request.json['11'])
  b12= float(request.json['12'])
  b13= float(request.json['13'])
  b14= float(request.json['14'])

  p1=[b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14]
  print(p1)      
  
  y=Logistic_Regression(p1)
  print("_--------------------------------------------_")
  if y== 0.0 :

        return jsonify("You are (not) expected to  CKD")
        
  elif y == 1.0 :
          
        return jsonify("You are expected to be  CKD")    
    
@app.route('/books4', methods=['POST'])
def add_get_books4():
  b1= float(request.json['1'])
  b2 = float(request.json['2'])
  b3= float(request.json['3'])
  b4= float(request.json['4'])
  b5= float(request.json['5'])
  b6= float(request.json['6'])
  b7=float( request.json['7'])
  b8=float( request.json['8'])
  p1=[b1,b2,b3,b4,b5,b6,b7,b8]
  print(p1)      
  y=AdaBoost(p1)
  print("_--------------------------------------------_")
  if y== 0:

        return jsonify("You are (not) expected to be diabetic")
        
  elif y == 1:
          
        return jsonify("You are expected to be diabetic")
    
#-------------------------------------------------------------------
@app.route('/books5', methods=['POST'])
def add_get_books5():

  b1= request.json['1']
  print(type(b1))
  y=parkinson1(b1)
  print("===================")
  print(y)
  print("===================")
  values(y)  
  if y == [0]:
          return jsonify("You are expected to healthy")
        
  elif y == [1]:    
          return jsonify("You are expected to parkinson") 
      
@app.route('/books6', methods=['POST'])
def add_get_books6():
 
  b1= request.json['1']
  print(type(b1))
  y=parkinson2(b1)
  print("===================")
  print(y)
  print("===================")
  values1(y)  
  if y == [0]:
          return jsonify("You are expected to healthy")
        
  elif y == [1]:    
          return jsonify("You are expected to parkinson")
      
@app.route('/books7', methods=['POST'])
def add_get_books7():
 
  b1= request.json['1']
  print(type(b1))
  y = covid(b1)
  values2(y)  
  if y == 0:
      
    return jsonify("You are expected to COVID-19")
        
  elif y == 1:    
    return jsonify("You are expected to Lung_Opacity")
    
  elif y == 2: 
      
    return jsonify("You are  expected to NORMAL")
    
  elif y == 3:
      
     return jsonify("You are expected to Viral Pneumonia")  

@app.route('/books8', methods=['POST'])
def add_get_books8():
  b1= request.json['1']
  print(type(b1))
  y=Breast_Cancer(b1)
  print("===================")
  print(y)
  print("===================")
  values3(y)  
  if y == [0]:
          return jsonify("You are expected to healthy")
        
  elif y == [1]:    
          return jsonify("You are expected to Breast_Cancer")      
      
@app.route('/books9', methods=['POST'])
def add_get_books9():
  b1= request.json['1']
  print(type(b1))
  y= alzheimers(b1)
  print("===================")
  print(y)
  print("===================")
  values4(y)  
  if y == [0]:
    return jsonify("MildDemented")
     
  elif y == [1]:
     return jsonify("ModerateDemented")
     
  elif y == [2]:
      return jsonify("NonDemented")
  elif y == [3]:

      return jsonify("VeryMildDemented")    

@app.route('/books10', methods=['POST'])
def add_get_books10():
  b1= request.json['1']
  print(type(b1))
  y=skin(b1)
  print("===================")
  print(y)
  print("===================")
  values5(y)  
  if y == [0]:
          return jsonify("You are expected to benign")
        
  elif y == [1]:    
          return jsonify("You are expected to malignant")  
      
@app.route('/books11', methods=['POST'])
def add_get_books11():
  b1= request.json['1']
  print(type(b1))
  y= brain_tumor(b1)
  print("===================")
  print(y)
  print("===================")
  values6(y)  
  if y == [0]:
    return jsonify("glioma_tumor")
     
  elif y == [1]:
     return jsonify("no_tumor")
     
  elif y == [2]:
      return jsonify("meningioma_tumor")
  elif y == [3]:

      return jsonify("pituitary_tumor")        
        
if __name__ =='__main__':
    app.run(debug=(False))       