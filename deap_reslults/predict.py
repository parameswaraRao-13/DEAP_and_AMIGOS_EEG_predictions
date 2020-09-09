from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from pickle import load
import matplotlib.pyplot as plt
mo = load_model("W:\\param's direc\\EGG\\Deap_colab\\egg_0_valance.h5")
i_data=pd.read_csv("W:\\param's direc\\EGG\Deap_colab\\deap_egg_tuning\\test_data_100.csv")

x=i_data.values

sss=load(open("W:\param's direc\EGG\Deap_colab\deap_egg_tuning\scaler.pkl","rb"))

x_v=sss.transform(x)

x_v = np.reshape(x_v, (x_v.shape[0],1,1,x_v.shape[1]))

# ooo = mo.predict_classes(x_v[0])
# # output_list.append(ooo)
# print(ooo)

output_list=[]
for i in range(0,len(x)):
    ooo = mo.predict_classes(x_v[i])
    output_list.append(int(ooo[0]))
index=np.arange(len(output_list))
plt.bar(index,output_list)
plt.xlabel('Samples')
plt.ylabel('Valance')
plt.title('Valance Prediction from DEAP EEG data')
plt.show()