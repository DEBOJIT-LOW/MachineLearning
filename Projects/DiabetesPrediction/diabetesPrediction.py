import numpy as np
import pickle
load = pickle.load(open(
    "D:/DEBOJIT/Machine_Learning/Projects/DiabetesPrediction/diabetes_model.sav", 'rb'))
input_data = (7, 187, 68, 39, 304, 37.7, 0.254, 41)
input_data = np.asarray(input_data).reshape(1, -1)

prediction = load.predict(input_data)
print(prediction[0])
if (prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")
