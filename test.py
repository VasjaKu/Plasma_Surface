import matplotlib.pyplot as plt
import numpy as np
from math import pi as pi
import math as m
import matplotlib as mpl

#cmapBO = mpl.colors.LinearSegmentedColormap.from_list(
#        'unevently divided', [(0, 'b'), (.5, 'gray'), (1, 'orange')])
# Prepare arrays x, y, z
e, k, delta, A = float(input()), float(input()), float(input()), float(input())
alpha = m.asin(delta)
x1 = 1 + e
x2 = 1 - e
x3 = 1 - delta * e
y1 = 0
y2 = k * e
N1 = -(1 + alpha) * (1 + alpha) / (e * k * k)
N2 = (1 - alpha) * (1 - alpha) / (e * k * k)
N3 = - k / (e * np.cos(alpha) * np.cos(alpha))
left_side = np.array([[1, x1**2, (y1**2-x1**2*np.log(x1)), (x1**4-4*x1**2*y1**2), (2*y1**4-9*x1**2*y1**2+3*x1**4*np.log(x1)-12*x1**2*y1**2*np.log(x1)), (x1**6-12*x1**4*y1**2+8*x1**2*y1**4), (8*y1**6-140*y1**4*x1**2+75*y1**2*x1**4-15*x1**6*np.log(x1)+180*x1**4*y1**2*np.log(x1)-120*x1**2*y1**4*np.log(x1))],
                     [1, x2**2, (y1**2-x2**2*np.log(x2)), (x2**4-4*x2**2*y1**2), (2*y1**4-9*x2**2*y1**2+3*x2**4*np.log(x2)-12*x2**2*y1**2*np.log(x2)), (x2**6-12*x2**4*y1**2+8*x2**2*y1**4), (8*y1**6-140*y1**4*x2**2+75*y1**2*x2**4-15*x2**6*np.log(x2)+180*x2**4*y1**2*np.log(x2)-120*x2**2*y1**4*np.log(x2))],
                     [1, x3**2, (y2**2-x3**2*np.log(x3)), (x3**4-4*x3**2*y2**2), (2*y2**4-9*x3**2*y2**2+3*x3**4*np.log(x3)-12*x3**2*y2**2*np.log(x3)), (x3**6-12*x3**4*y2**2+8*x3**2*y2**4), (8*y2**6-140*y2**4*x3**2+75*y2**2*x3**4-15*x3**6*np.log(x3)+180*x3**4*y2**2*np.log(x3)-120*x3**2*y2**4*np.log(x3))],
                     [0, 2*x3, (-2*x3*np.log(x3)-x3), (4*x3**3-8*x3*y2**2), (-30*x3*y2**2+12*x3**3*np.log(x3)+3*x3**3-24*x3*y2**2*np.log(x3)), (6*x3**5-48*x3**3*y2**2+16*x3*y2**4), (-400*x3*y2**4+480*x3**3*y2**2-90*x3**5*np.log(x3)-15*x3**5+720*x3**3*y2**2*np.log(x3)-240*x3*y2**4*np.log(x3))],
                     [0, - N1 * 2*x1, - N1 * (-2*x1*np.log(x1)-x1) - 2, - N1 * (4*x1**3-8*x1*y1**2) - (-8*x1**2), - N1 * (-30*x1*y1**2+12*x1**3*np.log(x1)+3*x1**3-24*x1*y1**2*np.log(x1)) - (24*y1**2-18*x1**2-24*x1**2*np.log(x1)), - N1 * (6*x1**5-48*x1**3*y1**2+16*x1*y1**4) - (-24*x1**4+96*x1**2*y1**2), - N1 * (-400*x1*y1**4+480*x1**3*y1**2-90*x1**5*np.log(x1)-15*x1**5+720*x1**3*y1**2*np.log(x1)-240*x1*y1**4*np.log(x1)) - (240*y1**4-1680*x1**2*y1**2+150*x1**4+360*x1**4*np.log(x1)-1440*x1**2*y1**2*np.log(x1))],
                     [0, - N2 * 2*x2, - N2 * (-2*x2*np.log(x2)-x2) - 2, - N2 * (4*x2**3-8*x2*y1**2) - (-8*x2**2), - N2 * (-30*x2*y1**2+12*x2**3*np.log(x2)+3*x2**3-24*x2*y1**2*np.log(x2)) - (24*y1**2-18*x2**2-24*x2**2*np.log(x2)), - N2 * (6*x2**5-48*x2**3*y1**2+16*x2*y1**4) - (-24*x2**4+96*x2**2*y1**2), - N2 * (-400*x2*y1**4+480*x2**3*y1**2-90*x2**5*np.log(x2)-15*x2**5+720*x2**3*y1**2*np.log(x2)-240*x2*y1**4*np.log(x2)) - (240*y1**4-1680*x2**2*y1**2+150*x2**4+360*x2**4*np.log(x2)-1440*x2**2*y1**2*np.log(x2))],
                     [0, - 2, - N3 * (2*y2) - (-2*np.log(x3)-3), - N3 * (-8*x3**2*y2) - (12*x3**2-8*y2**2), - N3 * (8*y2**3-18*x3**2*y2-24*x3**2*y2*np.log(x3)) - (-54*y2**2+36*x3**2*np.log(x3)+21*x3**2-24*y2**2*np.log(x3)), - N3 * (32*x3**2*y2**3-24*x3**4*y2) - (30*x3**4-144*x3**2*y2**2+16*y2**4), - N3 * (150*x3**4*y2-560*x3**2*y2**3+360*x3**4*y2*np.log(x3)-480*x3**2*y2**3*np.log(x3)+48*y2**5) - (-640*y2**4+2160*x3**2*y2**2-450*x3**4*np.log(x3)-165*x3**4+2160*x3**2*y2**2*np.log(x3)-240*y2**4*np.log(x3))]])
right_side = np.array([-1 * ((1 - A) * pow(x1, 4) / 8 + A * pow(x1, 2) * np.log(x1) / 2),
                       -1 * ((1 - A) * pow(x2, 4) / 8 + A * pow(x2, 2) * np.log(x2) / 2),
                       -1 * ((1 - A) * pow(x3, 4) / 8 + A * pow(x3, 2) * np.log(x3) / 2),
                       -1 * ((1 - A) * pow(x3, 3) / 2 + A * x3 * np.log(x3) + A * x3 / 2),
                       N1 * ((1 - A) * pow(x1, 3) / 2 + A * x1 * np.log(x1) + A * x1 / 2),
                       N2 * ((1 - A) * pow(x2, 2) / 2 + A * x2 * np.log(x2) + A * x2 / 2),
                       3 * (1 - A) * pow(x3, 2) / 2 + A * np.log(x3) + A * 3 / 2])
const = np.linalg.inv(left_side). dot(right_side)
t = np.linspace(0, 2*pi, 100)
a = m.asin(delta)
x = 1 + e * np.cos(t + a * np.sin(t))
y = e * k * np.sin(t)
X, Y = np.meshgrid(x, y)
psi = (1-A)*X**4/8+A*X**2*np.log(X)/2+1*const[0]+X**2*const[1]+(Y**2-X**2*np.log(X))*const[2]+(X**4-4*X**2*Y**2)*const[3]+(2*Y**4-9*X**2*Y**2+3*X**4*np.log(X)-12*X**2*Y**2*np.log(X))*const[4]+(X**6-12*X**4*Y**2+8*X**2*Y**4)*const[5]+(8*Y**6-140*Y**4*X**2+75*Y**2*X**4-15*X**6*np.log(X)+180*X**4*Y**2*np.log(X)-120*X**2*Y**4*np.log(X))*const[6]
pmin, pmax = np.min(psi), np.max(psi)

plt.figure(figsize=(4, 4), dpi=90)
plt.xlabel('R/R0')
plt.ylabel('Z/R0')
plt.title('Поверхности полоидального тока\n' + r'A = %g, $\epsilon$ = %g, k = %g, $\delta$ = %g' % (A, e, k, delta),
          fontsize=16, color='black')

lev = 20  # psi levels number
psurf = 0  # set psi limit to plasma boundary
levels_in = np.linspace(pmin, psurf, lev + 1)
# colors = plt.cm.jet(np.linspace(0, 1, lev+1)) #
cont_in = plt.contour(X, Y, psi, levels_in, linewidths=1, cmap=plt.cm.jet)
#cont_filled = plt.contourf(X, Y, psi, levels_in, cmap=cmapBO)
#levels_out = np.linspace(psurf, pmax, lev + 1)
#cont_out = plt.contour(X, Y, psi, levels_out, linewidths = 1, cmap = plt.cm.Greys_r)
plt.grid()
plt.show()
