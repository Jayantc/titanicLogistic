import joblib
import numpy as np
print('Enter your details to find whether you could had survived or not in titanic crash')
age= int(input('Enter your age : '))
fare= int(input('Enter ticket price between 5 to 520 : '))
if fare>520 or fare<5:
    print('wrong input')
    exit()
gender=input('Enter your gender male or female : ')
if gender=='male':
    gender=0
elif gender=='female':
    gender=1
else:
    print('Wrong input')
    exit()
pclass= int(input('Enter your Passenger Class 1, 2 or 3 : '))
if pclass==1:
    pclass1, pclass2= 0, 0
elif pclass==2:
    pclass1, pclass2= 1, 0
elif pclass==3:
    pclass1, pclass2= 0, 1
else:
    print('Wrong input')
    exit()

model= joblib.load('TitanicModel')
data= np.array([[age, fare, gender, pclass1, pclass2]])

predict= model.predict(data)

if predict == [1]:
    print("Congratulations!!! You have survived in titanic crash!")
else:
    print("Sorry! You have not survived in titanic crash!")