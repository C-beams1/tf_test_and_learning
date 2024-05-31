import numpy
import time



pos = 0
mubiao = 1
dt = 0.2
v = 0
vmax = 10
err = 0
kp, ki, kd = 10, 1, 10
kis = 0
err1 = 0
vs = 100
poss = []

for i in range(1000):
    err = mubiao - pos
    kis += err
    v = vmax * (err * kp + ki * kis / ((i + 1) * dt) + kd * (err - err1)) / vs
    if v > vmax:
        v = vmax
    err1 = err
    print('{:.2f}'.format(v))
    pos += v * dt
    # r = numpy.random.random(1)
    # pos -= 0.1 * r[0]
    poss.append(pos)

for i in poss:
    print('{:.2f}'.format(i), end=' ')
# print(r)
