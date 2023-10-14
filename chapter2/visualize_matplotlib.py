import matplotlib.pyplot as plt 
import numpy as np 

# PIE chart plotting
colors = ['orange','green','cyan','skyblue','yellow','red','blue','white','black','pink']

dict_top10 = {"":""}
# pie chart plot 
plt.pie(list(dict_top10.values()), labels=dict_top10.keys(),colors=colors, autopct='%2.1f%%', shadow=True, startangle=90)

# 暫時真實的plot
plt.show()

# bar chart plotting
x_states = dict_top10.keys()
y_vaccine_dist_1 = dict_top10.values()

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
# bar values filling with x-axis /y-axis values
ax.bar(np.arange(len(x_states)),y_vaccine_dist_1,log=1)
plt.show()
