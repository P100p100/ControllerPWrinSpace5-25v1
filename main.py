import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
I = np.array([[32.99,0,0],[0,778.8,0],[0,0,778.8]])
def Torque(angles):
    az = angles[0]
    bz = angles[1]
    ay = angles[2]
    by = angles[3]
    Tz1 = 0.000508 * np.sin(3 * bz) + 0.000653 * np.cos(2 * bz) * np.sin(3 * az) + 0.000125 * np.cos(3 * az) * np.sin(3 * bz)
    Tz2 = 0.00834 * np.cos(4 * az) - 0.00452 * np.cos(4 * bz) - 0.00258 * np.cos(4 * bz) * np.sin(4 * az)
    Tz3 = 0.189 * np.sin(2 * bz) - 0.0277 * np.sin(4 * az) - 0.00607 * np.cos(4 * az) - 0.269 * np.sin(bz)
    Ty1 = 0.000508 * np.sin(3 * by) + 0.000653 * np.cos(2 * by) * np.sin(3 * ay) + 0.000125 * np.cos(3 * ay) * np.sin(3 * by)
    Ty2 = 0.00834 * np.cos(4 * ay) - 0.00452 * np.cos(4 * by) - 0.00258 * np.cos(4 * by) * np.sin(4 * ay)
    Ty3 = 0.189 * np.sin(2 * by) - 0.0277 * np.sin(4 * ay) - 0.00607 * np.cos(4 * ay) - 0.269 * np.sin(by)
    return(np.array([[Tz1+Ty1],[Tz2+Ty3],[Tz3-Ty2]]))
def findAngles(desiredTorque,R):
    Topt = desiredTorque
    def F(angles):
        T = Torque(angles)
        return ((R**2*(T[0, 0] - Topt[0, 0]) ** 2 + (T[1, 0] - Topt[1, 0]) ** 2 + (T[2, 0] - Topt[2, 0]) ** 2) ** 0.5)
    Q = sp.optimize.minimize(F, np.array([0, 0, 0, 0])).x
    return(Q,Torque(Q))
def angular_acc(T):
    return(np.array([[T[0,0]],[T[1,0]],[T[2,0]]]))
def Torq(Add):
    return (np.array([[Add[0, 0]], [Add[1, 0]], [Add[2, 0]]]))
def StateTimeDerivative(S,ang_acc):
    return(np.array([[S[3,0]],[S[4,0]],[S[5,0]],[ang_acc[0,0]],[ang_acc[1,0]],[ang_acc[2,0]]]))
def stateIncrement(S,ang_acc,dt):
    k1 = StateTimeDerivative(S, ang_acc)
    k2 = StateTimeDerivative(S + k1 * dt / 2, ang_acc)
    k3 = StateTimeDerivative(S + k2 * dt / 2, ang_acc)
    k4 = StateTimeDerivative(S + k3 * dt, ang_acc)
    return (dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
def desiredAdd(S,W,dt,max_yz):
    dAdd = np.array([[-S[0,0]/dt**2*W[0,0]-S[3,0]/dt*W[0,1]],[-S[1,0]/dt**2*W[1,0]-S[4,0]/dt*W[1,1]],[-S[2,0]/dt**2*W[2,0]-S[5,0]/dt*W[2,1]]])
    if dAdd[0,0]>0.0002:
        dAdd[0,0] = 0.0002
    if dAdd[0,0]<-0.0002:
        dAdd[0,0] = -0.0002
    if dAdd[1,0]>max_yz:
        dAdd[1,0] = max_yz
    if dAdd[1,0]<-max_yz:
        dAdd[1,0] = -max_yz
    if dAdd[2,0]>max_yz:
        dAdd[2,0] = max_yz
    if dAdd[2,0]<-max_yz:
        dAdd[2,0] = -max_yz
    return(dAdd)
duration = 6
Tconst = 2.645
deltaT = 0.1
weights = np.array([[0,0.1],[0.4,1],[0.4,1]])
weightX = 10
Dt = 0.1
state = np.array([[0],[0.04],[0.001],[0.000004],[0.02],[-0.1]])
state = np.array([[0],[0.04*(random.random()-1/2)*2],[0.1*(random.random()-1/2)*2],[0.0003*(random.random()-1/2)*2],[0.05*(random.random()-1/2)*2],[0.05*(random.random()-1/2)*2]])
X = []
Y = []
Z = []
VX = []
VY = []
VZ = []
AX = []
AY = []
AZ = []
T = []
timesince = deltaT
for i in range(int(duration/Dt)):
    t = i*Dt
    Tcoeff = Tconst*(6-t)**2
    if timesince>=deltaT:
        angacc = angular_acc(Torque(findAngles(Torq(desiredAdd(state,weights,deltaT,0.05)),weightX)[0]))
        timesince += -deltaT
    angacc = angacc + np.array([[0.0002*(random.random()-1/2)*2],[0.05*(random.random()-1/2)*2],[0.05*(random.random()-1/2)*2]])*0
    state = state + stateIncrement(state,angacc,Dt)
    timesince += Dt
    T.append(t)
    X.append(state[0,0])
    Y.append(state[1, 0])
    Z.append(state[2, 0])
    VX.append(state[3, 0])
    VY.append(state[4, 0])
    VZ.append(state[5, 0])
    AX.append(angacc[0, 0])
    AY.append(angacc[1, 0])
    AZ.append(angacc[2, 0])
plt.plot(T,X)
plt.plot(T,VX)
plt.plot(T,AX)
plt.show()
plt.plot(T,Y)
plt.plot(T,VY)
plt.plot(T,AY)
plt.show()
plt.plot(T,Z)
plt.plot(T,VZ)
plt.plot(T,AZ)
plt.show()