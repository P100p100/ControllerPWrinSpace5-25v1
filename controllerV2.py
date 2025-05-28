import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
I = np.array([[32.99,0,0],[0,778.8,0],[0,0,778.8]])
hI = 0.713
aT = 1.5
m = 3.5
ro = 1.1225
v0 = 60
g = 10
fi = g/v0
A = 0.05**2*np.pi
f0 = ro*v0**2*A/2
Icg = np.array([[32.99,0,0],[0,778.8-m*hI**2,0],[0,0,778.8-m*hI**2]])
IcgInv = sp.linalg.inv(Icg)
weights = np.array([[0,1],[0.2,1],[0.2,1]])
weights2 = np.array([[0,1],[1,1],[1,1]])
weightX = 50
Cmax = np.array([[0.0003],[0.1],[0.1]])
delT = 0.01
dt = 0.01
duration = 1/fi
def TorqueS(angles):
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
def ForceS(angles):
    az = angles[0]
    bz = angles[1]
    ay = angles[2]
    by = angles[3]
    Fz2 = 0.00588 * np.cos(2 * az) + 0.1 * np.sin(bz) - 0.103 * np.cos(bz) * np.sin(az)
    Fz3 = 0.0358 * np.cos(2 * bz) - 0.0314 * np.cos(2 * az) - 0.00335 * np.cos(2 * bz) * np.cos(2 * az)
    Fy2 = 0.00588 * np.cos(2 * ay) + 0.1 * np.sin(by) - 0.103 * np.cos(by) * np.sin(ay)
    Fy3 = 0.0358 * np.cos(2 * by) - 0.0314 * np.cos(2 * ay) - 0.00335 * np.cos(2 * by) * np.cos(2 * ay)
    return(np.array([[0],[Fz2+Fy3],[Fz3-Fy2]]))
def TorqueCoeffs(angles):
    T = TorqueS(angles)
    F = ForceS(angles)
    return(np.array([[T[0,0]*aT],[T[1,0]*aT-F[2,0]*hI],[T[2,0]*aT+F[1,0]*hI]]))
def findAngles(TCopt):
    def Fcost(angles):
        T = TorqueCoeffs(angles)
        return ((weightX**2*(T[0, 0] - TCopt[0, 0]) ** 2 + (T[1, 0] - TCopt[1, 0]) ** 2 + (T[2, 0] - TCopt[2, 0]) ** 2) ** 0.5)
    Q = sp.optimize.minimize(Fcost, np.array([0, 0, 0, 0])).x
    return(Q)
def AddFromTC(TC,t):
    return (np.matmul(IcgInv, TC)*f0*(1-fi*t)**2)
def StateTimeDerivative(S,add):
    return(np.array([[S[3,0]],[S[4,0]],[S[5,0]],[add[0,0]],[add[1,0]],[add[2,0]]]))
def stateIncrement(S,add,dt):
    k1 = StateTimeDerivative(S, add)
    k2 = StateTimeDerivative(S + k1 * dt / 2, add)
    k3 = StateTimeDerivative(S + k2 * dt / 2, add)
    k4 = StateTimeDerivative(S + k3 * dt, add)
    return (dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
def desiredCoeffs(S,t):
    desC = np.array([[0], [0], [0]])
    desAdd = np.array([[0], [0], [0]])
    AddMax = np.matmul(IcgInv,Cmax)*f0*(1-fi*t)**2
    #Zero conditions
    A0 = np.array([[0], [delT ** 2 * weights2[1, 0] * AddMax[1, 0]], [delT ** 2 * weights2[2, 0] * AddMax[2, 0]]])
    Ad0 = np.array([[delT * weights2[0, 1] * AddMax[0, 0]], [delT * weights2[1, 1] * AddMax[1, 0]], [delT * weights2[2, 1] * AddMax[2, 0]]])
    def AP(index):
        M = np.matmul(IcgInv,Cmax)[index,0]*f0*np.sign(S[index,0])
        Tk = np.roots(np.array([fi**2*M/3,-fi*M,M,S[index+3,0]+M*(-t+fi*t**2-fi**2*t**3/3)]))
        tk = 0
        for i in Tk:
            if np.isreal(i):
                tk = i
        if tk<=t or tk>1/fi:
            return(-S[index,0])
        else:
            return(S[index,0]+(tk-t)*(S[index+3,0]+M*(-t+fi*t**2-fi**2*t**3/3))+M/2*(tk**2-t**2)-fi*M/3*(tk**3-t**3)+fi**2*M/12*(tk**4-t**4))
    #yaw
    if S[1,0]>A0[1,0]:
        if S[4,0]>=-Ad0[1,0]:
            desAdd = np.array([[desAdd[0,0]],[-AddMax[1,0]],[desAdd[2,0]]])
        else:
            #APcheck
            Ap = AP(1)
            if Ap>0:
                desAdd = np.array([[desAdd[0, 0]], [-AddMax[1, 0]], [desAdd[2, 0]]])
            else:
                desAdd = np.array([[desAdd[0, 0]], [AddMax[1, 0]], [desAdd[2, 0]]])
    elif S[1, 0] < -A0[1, 0]:
        if S[4, 0] <= Ad0[1, 0]:
            desAdd = np.array([[desAdd[0, 0]], [AddMax[1, 0]], [desAdd[2, 0]]])
        else:
            # APcheck
            Ap = AP(1)
            if Ap < 0:
                desAdd = np.array([[desAdd[0, 0]], [AddMax[1, 0]], [desAdd[2, 0]]])
            else:
                desAdd = np.array([[desAdd[0, 0]], [-AddMax[1, 0]], [desAdd[2, 0]]])
    else:
        if S[4, 0] < -Ad0[1, 0]:
            desAdd = np.array([[desAdd[0, 0]], [AddMax[1, 0]], [desAdd[2, 0]]])
        elif S[4,0] > Ad0[1,0]:
            desAdd = np.array([[desAdd[0,0]],[-AddMax[1,0]],[desAdd[2,0]]])
        else:
            desAdd = np.array([[desAdd[0, 0]], [-S[4, 0] / delT * weights[1, 1] -S[1, 0] / delT**2 * weights[1, 0]], [desAdd[2, 0]]])
    #pitch
    if S[2,0]>A0[2,0]:
        if S[5,0]>=-Ad0[2,0]:
            desAdd = np.array([[desAdd[0,0]],[desAdd[1,0]],[-AddMax[2,0]]])
        else:
            # APcheck
            Ap = AP(2)
            if Ap > 0:
                desAdd = np.array([[desAdd[0,0]],[desAdd[1,0]],[-AddMax[2,0]]])
            else:
                desAdd = np.array([[desAdd[0,0]],[desAdd[1,0]],[AddMax[2,0]]])
    elif S[2, 0] < -A0[2, 0]:
        if S[5, 0] <= Ad0[2, 0]:
            desAdd = np.array([[desAdd[0, 0]], [desAdd[1, 0]], [AddMax[2, 0]]])
        else:
            # APcheck
            Ap = AP(2)
            if Ap < 0:
                desAdd = np.array([[desAdd[0, 0]], [desAdd[1, 0]], [AddMax[2, 0]]])
            else:
                desAdd = np.array([[desAdd[0, 0]], [desAdd[1, 0]], [-AddMax[2, 0]]])
    else:
        if S[5, 0] < -Ad0[2, 0]:
            desAdd = np.array([[desAdd[0, 0]], [desAdd[1, 0]], [AddMax[2, 0]]])
        elif S[5,0] > Ad0[2,0]:
            desAdd = np.array([[desAdd[0,0]],[desAdd[1,0]],[-AddMax[2,0]]])
        else:
            desAdd = np.array([[desAdd[0, 0]], [desAdd[1, 0]], [-S[5, 0] / delT * weights[2, 1] -S[2, 0] / delT**2 * weights[2, 0]]])
    #roll
    if S[3,0]>Ad0[0,0]:
        desAdd = np.array([[-AddMax[0,0]],[desAdd[1,0]],[desAdd[2,0]]])
    elif S[3,0]<-Ad0[0,0]:
        desAdd = np.array([[AddMax[0,0]],[desAdd[1,0]],[desAdd[2,0]]])
    else:
        desAdd = np.array([[-S[3,0]/delT*weights[0,1]],[desAdd[1,0]],[desAdd[2,0]]])
    #ADD--->C
    desC = np.matmul(Icg,desAdd)/f0/(1-fi*t)**2
    for i in range(3):
        if desC[i,0]>Cmax[i,0]:
            desC[i, 0] = Cmax[i, 0]
        if desC[i,0]<-Cmax[i,0]:
            desC[i, 0] = -Cmax[i, 0]
    return(desC)
state = np.array([[0],[0.00003],[-0.0004],[0.0001],[0],[0]])
state = np.array([[0],[0.002*(random.random()-1/2)*2],[0.002*(random.random()-1/2)*2],[0.0002*(random.random()-1/2)*2],[0.002*(random.random()-1/2)*2],[0.002*(random.random()-1/2)*2]])
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
timesince = delT
C = np.array([[0],[0],[0]])
for i in range(int(duration/dt)):
    t = i*dt
    if timesince>=delT:
        C = TorqueCoeffs(findAngles(desiredCoeffs(state,t)))
        timesince += -delT
    ADD = np.matmul(IcgInv,C)*f0*(1-fi*t)**2
    state = state + stateIncrement(state,ADD,dt)
    timesince += dt
    T.append(t)
    X.append(state[0,0])
    Y.append(state[1, 0])
    Z.append(state[2, 0])
    VX.append(state[3, 0])
    VY.append(state[4, 0])
    VZ.append(state[5, 0])
    AX.append(ADD[0, 0])
    AY.append(ADD[1, 0])
    AZ.append(ADD[2, 0])
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