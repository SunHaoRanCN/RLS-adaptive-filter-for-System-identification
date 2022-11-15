import numpy as np
import matplotlib.pyplot as plt
import timeit
import pyroomacoustics as pra
from scipy.signal import chirp

#%% RLS

def RLS(x, y, M):
    
    n = len(x)
    
    I = np.eye(M)
    
    y_out = np.zeros(len(y))
    Eta_out = np.zeros(len(y))
    w_out = np.zeros(M)
    
    for i in range(n):
        if i == 0:  # First time
            P_last = I
            w_last = np.zeros((M, 1))
            
        d = y[i]  # Desired data vactor (Expected output)
        
        if i<M:
            xn = np.pad(x[i::-1], (0, M-i-1), 'constant', constant_values=(0, 0))
        else:
            xn = x[i:i-M:-1]
        xn = xn.reshape((M, 1))
        
        K = (P_last @ xn) / (lamda + xn.T @ P_last @ xn)  # Gain vactor
        yn = w_last.T @ xn  # Output of filter
        Eta = d-yn  # Error
        eta = np.abs(Eta)**2
        w = w_last + K @ Eta  # Coefficients
        P = (I - K @ xn.T) @ P_last/lamda  # Error correlation matrix
        
        P_last = P
        w_last = w
        
        y_out[i] = yn
        Eta_out[i] = eta
        w_out = w
        
    return y_out, w_out, Eta_out

#%% Averange

# def RE(n, M, N=10):
def RE(n, ht, M, SNR, N=10):
    
    # Y = np.zeros(n)
    # W = np.zeros((M, 1))
    # eta = np.zeros(n)
    
    Y = np.zeros(n+M-1)
    W = np.zeros((M, 1))
    eta = np.zeros(n+M-1)
    
    for i in range(N):
        
        # data = np.loadtxt(r'E:\M1_S2\Projet\zsProjetM1\measure{}\TemporalData.txt'.format(i+1), skiprows=1)
        
        # xt = data[:, 1]
        
        # yt = data[:, 2]

        # n = len(xt)

        # M = int(n/20)
        
        xt = np.random.randn(n)  # Gaussian white noise
        # t = np.arange(0, 1, 1/n)
        # xt = chirp(t, f0=100, t1=1, f1=1000)
        # xt = np.sin(2*np.pi*t)
        
        yt = np.convolve(xt, ht)
        
        sigpower = np.sum(np.abs(yt**2))/len(yt) #Power
        repSNR = 10**(SNR/10)  # Power ratio
        noisepower = sigpower/repSNR # power of noise
        noise = np.sqrt(noisepower/2)*np.random.randn(len(yt)) #+1j*np.random.randn(len(yt))
        
        yt = yt+noise
        
        out = RLS(xt, yt, M)
        
        Y = out[0]
        W+= out[1]
        eta+= out[2]
        
    W = W/N
    eta = eta/N
    
    return xt, yt, Y, W, eta

#%% IR

M = 100
# ht = np.random.randn(M)
ht = np.zeros(M)
ht[int(M/3)] = 0.8

#%%

t = np.linspace(0, 1, M)

n = 12800
lamda = 1  # forgetting factor
N = 1

SNR = 30

xt1, yt1, Y1, W1, ETA1 = RE(n, ht, M, SNR, N)

#%% 模拟画图

plt.close('all')
# plt.figure()
# plt.plot(yt1, label='Real output')
# plt.plot(Y1, '--', label='Output of RLS')
# plt.legend()
# plt.grid()
# plt.title('Output of filter')
# plt.show()

plt.figure()
plt.plot(t, ht, label='h (RI Théorie)')
plt.plot(t, W1, '--', label='W (RI calculé)')
plt.title('Comparasion des Réponses impulsionnelles', fontsize=15)
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.grid()
plt.legend(prop = {'size':15})
plt.show()

plt.figure()
plt.title('Learning curve', fontsize=15)
plt.semilogy(ETA1)
plt.xlabel("Nombre d'itérations", fontsize=15)
plt.ylabel('Erreurs (dB)', fontsize=15)
plt.grid()
plt.show()

#%% SNR-MIS

SNR = np.arange(0, 30, 1)

Mis = []
for i in SNR:
    xt, yt, Y, Wi, ETA = RE(n, ht, M, i, 1)
    Wi = Wi.flatten()
    mis = 10*np.log(np.linalg.norm(Wi-ht))
    Mis.append(mis)
    
plt.figure()
plt.title('Misalignment en fonction de RSB', fontsize=15)
plt.plot(SNR, Mis)
plt.xlabel('RSB', fontsize=15)
plt.ylabel('Misalignment (dB)', fontsize=15)
plt.grid()
plt.show()

#%% M-TIME
lamda = 1
n = 1000

M = np.arange(1, 100, 10) #lenth

time = []

for i in M:
    
    hti = np.zeros(i)
    hti[int(i/3)] = 0.8
    
    start = timeit.default_timer()
    xt, yt, Y, W, ETA = RE(n, hti, i, 30, 1)
    stop = timeit.default_timer()
    ti = stop-start
    time.append(ti)
#%%
plt.figure()
plt.title('Temps de calcul en fonction de M', fontsize=15)
plt.plot(M, time, label='temps de calcul')
plt.plot(M, M**2/100000+0.02, '--', label='M$^2$/1e5+0.02')
plt.xlabel('Longueur M', fontsize=15)
plt.ylabel('time (s)', fontsize=15)
plt.legend(prop = {'size':15})
plt.grid()
plt.show()

#%% Simulation

t = np.arange(0, 10, 0.00001)
x = chirp(t, f0=1000, f1=1, t1=10, method='linear')

plt.plot(t, x)
plt.title("Linear Chirp, f(0)=6, f(10)=1")
plt.xlabel('t (sec)')
plt.show()

#%% 一次测量求平均
lamda = 1

data = np.loadtxt(r'E:\M1_S2\Projet\ZSprojet\100\TemporalData.txt', skiprows=1)

xt = data[:, 1]

yt = data[:, 2]

n = len(xt)

M = 200

start = timeit.default_timer()
Y, W, ETA = RLS(xt, yt, M)
stop = timeit.default_timer()
print(stop-start)

#%%

E = []

L = 1000
for i in range(len(ETA)):
    if i<L:
        e = np.pad(ETA[i::-1], (0, L-i-1), 'constant', constant_values=(0, 0))
    else:
        e = ETA[i:i-L:-1]
    ee = np.sum((np.abs(e))**2)/L
    E.append(ee)
    
#%%

# with open('20ciY.txt', 'w+') as data:
#     for i in Y1:
#         tmp = '{}\r'.format(i)
#         data.writelines(tmp)
# W1 = W1.flatten()
# with open('20ciW.txt', 'w+') as data:
#     for i in W1:
#         tmp = '{}\r'.format(i)
#         data.writelines(tmp)
# with open('20ciETA.txt', 'w+') as data:
#     for i in ETA1:
#         tmp = '{}\r'.format(i)
#         data.writelines(tmp)

#%%

t = np.linspace(0, 200/12800, 200)

plt.figure()
plt.plot(t, W, label='W')
plt.title('Réponse impulsionnelle de la salle', fontsize=15)
plt.grid()
plt.xlabel('time (s)', fontsize=15)
plt.ylabel('Amplitude', fontsize=15)
plt.legend(prop = {'size':15})
plt.show()

plt.figure()
plt.title('Learning curve with calcul approximatif')
plt.semilogy(E)
plt.xlabel("Nombre d'itérations")
plt.ylabel('Erreurs (dB)')
plt.grid()
plt.show()

# plt.figure()
# plt.plot(yt)

# plt.figure()
# plt.plot(xt)

# plt.figure()
# plt.pcolormesh(P)

#%% 20个
lamda = 1

start = timeit.default_timer()
xt1, yt1, Y1, W1, ETA1 = RE(12800, 200, 10)
stop = timeit.default_timer()
print(stop-start)

#%%

# plt.figure()
# plt.plot(yt1, label='Real output')
# plt.plot(Y1, '--', label='Output of RLS')
# plt.legend()
# plt.grid()
# plt.title('Output of filter')
# plt.show()

t = np.linspace(0, 200/12800, 200)

plt.figure()
plt.plot(t, W1, label='W')
plt.title('Réponse impulsionnelle de la salle')
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()

plt.figure()
plt.title('Learning curve with 10 moyenne')
plt.semilogy(ETA1)
plt.xlabel("Nombre d'itérations")
plt.ylabel('Erreurs (dB)')
plt.grid()
plt.show()

#%% 时变系统

def TVS(x, y, M):
    
    n = len(x)
    
    I = np.eye(M)
    
    y_out = np.zeros(len(y))
    Eta_out = np.zeros(len(y))
    w_out = np.zeros(M)
    W_k = []
    
    for i in range(n):
        if i == 0:  # First time
            P_last = I
            w_last = np.zeros((M, 1))
            
        d = y[i]  # Desired data vactor (Expected output)
        
        if i<M:
            xn = np.pad(x[i::-1], (0, M-i-1), 'constant', constant_values=(0, 0))
        else:
            xn = x[i:i-M:-1]
        xn = xn.reshape((M, 1))
        
        K = (P_last @ xn) / (lamda + xn.T @ P_last @ xn)  # Gain vactor
        yn = w_last.T @ xn  # Output of filter
        Eta = d-yn  # Error
        eta = Eta**2
        w = w_last + K @ Eta  # Coefficients
        P = (I - K @ xn.T) @ P_last/lamda  # Error correlation matrix
        
        P_last = P
        w_last = w
        
        y_out[i] = yn
        Eta_out[i] = eta
        w_out = w
        W_k.append(w)
        
    return y_out, w_out, W_k, Eta_out

#%%

lamda = 0.98

data = np.loadtxt(r'E:\M1_S2\Projet\zsProjetM1\bouge2\TemporalData.txt', skiprows=1)

xt = data[:, 1]

yt = data[:, 2]

Y, W, WK, ETA = TVS(xt, yt, 150)
# Y, W, ETA = RLS(xt, yt, 200)

WK = np.array(WK)
WK = WK.reshape((128000, 150)).T

#%%

plt.figure()
plt.pcolormesh(WK)
plt.xlabel("Nombre d'itérations (n)")
plt.ylabel('Longueur (K)')
plt.title('Réponse impulsionnelle des systèmes variant du temps ($\lambda$={})'.format(lamda))
plt.show()

#%%

E = []

L = 2000
for i in range(len(ETA)):
    if i<L:
        e = np.pad(ETA[i::-1], (0, L-i-1), 'constant', constant_values=(0, 0))
    else:
        e = ETA[i:i-L:-1]
    ee = np.sum((np.abs(e))**2)/L
    E.append(ee)

#%%

plt.figure()
plt.title('Learning curve')
plt.semilogy(E)
plt.xlabel("Nombre d'itérations")
plt.ylabel('Erreurs (dB)')
plt.grid()
plt.show()