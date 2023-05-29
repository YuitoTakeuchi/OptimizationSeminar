import matplotlib.pyplot as plt
import numpy as np
import pathlib

def obj_func(xx, yy):
    return (1.0-xx)**2 + (1.0-yy)**2 + 0.5*(2.0*yy-xx**2)**2

# X座標
x = np.linspace(-3, 3, 500)
y = np.linspace(-3, 3, 500)
# 格子点作成
x, y = np.meshgrid(x, y)
z = obj_func(x, y)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.contour(x,y,z,np.linspace(0,160,20),colors='black') 
ax.contourf(x,y,z,np.linspace(0,160,40),cmap='Blues')
plt.gca().set_aspect('equal') # アスペクト比を1:1に


folder_path = str(pathlib.Path(__file__).parent) + "/result"

# Gradient Descent
xs = []
ys = []
with open(folder_path + "/gradient_descent.txt") as f:
    while(1):
        data = list(map(float, f.readline().split()))
        if len(data) < 3:
            break
        xs.append(data[0])
        ys.append(data[1])
ax.plot(xs, ys, label="Gradient Descent", color="magenta")

# Conjugate Gradient
xs = []
ys = []
with open(folder_path + "/conjugate_gradient.txt") as f:
    while(1):
        data = list(map(float, f.readline().split()))
        if len(data) < 3:
            break
        xs.append(data[0])
        ys.append(data[1])
ax.plot(xs, ys, label="Conjugate Gradient", color="lime")

# Newton's Method
xs = []
ys = []
with open(folder_path + "/newtons_method.txt") as f:
    while(1):
        data = list(map(float, f.readline().split()))
        if len(data) < 3:
            break
        xs.append(data[0])
        ys.append(data[1])
ax.plot(xs, ys, label="Newton's Method", color="r")


# BFGS Method
xs = []
ys = []
with open(folder_path + "/BFGS.txt") as f:
    while(1):
        data = list(map(float, f.readline().split()))
        if len(data) < 3:
            break
        xs.append(data[0])
        ys.append(data[1])
ax.plot(xs, ys, label="BFGS", color="navy")

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
        

