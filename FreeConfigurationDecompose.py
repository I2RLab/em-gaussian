import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('ColumbusCircle_Zero.xlsx')

X = df['X'] * 0.3048
# X = X.tolist()
# X = X * 0.3048

Y = df['y'] * 0.3048
# Y = Y.tolist()
# Y = Y * 0.3048

ID = df['Building_ID']
ID = ID.tolist()

del df

filename = 'Manhattan.poly'
file = open(filename, "w")
file.writelines(str(len(X))+' 2 1\n')

for i in range(len(X)):
	file.writelines(str(i + 1)+' '+str(X[i])+' '+str(Y[i])+'\n')

file.write(str(len(X))+' 0\n')

start_point = 1
for i in range(len(X)):
	# print i
	if  i  < len(X) - 1:
		if ID[i] == ID[i+1]:
			file.write(str(i+1)+' '+str(i+1)+' '+str(i+2) + '\n')
		else:
			file.write(str(i+1) + ' ' + str(i+1) + ' ' + str(start_point) + '\n')
			start_point = i + 2
			
	else:
		file.write(str(i+1) + ' ' + str(i+1) + ' ' + str(start_point) + '\n')
	
file.write(str(max(ID)) + '\n')

j = ID.count(0)
for i in range(1, max(ID) + 1):
	num_points = ID.count(i)
	x_mid = sum(X[j:j + num_points]) / num_points
	y_mid = sum(Y[j:j + num_points]) / num_points
	j += num_points
	file.write(str(i) + ' ' + str(x_mid) + ' ' + str(y_mid) + '\n')



file.close()
for i in range(max(ID)):
	print(ID.count(i))

plt.scatter(X, Y, marker = 'h', linewidths = 0.01)
plt.show()





