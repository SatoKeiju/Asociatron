import numpy as np
import matplotlib.pyplot as plt


mu = int(input('記憶するベクトルの数: '))
N = int(input('各ベクトルの行数: '))
noise = int(input('ノイズ発生確率(%): '))
noise_per = noise / 100
calc_times = 10


#S, n, xベクトル生成
S = np.random.choice([-1, 1], mu * N).reshape(N, mu)# / np.sqrt(N)
print('S')
print(S)

s_mu = []
for i in range(mu):
    s_mu.append(S[:,i].reshape(N, 1))

selected_num = np.random.randint(mu)
selected_data = s_mu[selected_num]
print('選ばれたデータ')
print(selected_data)

n = (np.random.choice([-1, 1], N, p = [noise_per, 1-noise_per])).reshape(N, 1)
#n = (np.random.choice([-1, 1], N)).reshape(N, 1)
##n1 = np.random.randint(-1, 1, (N, 1))
##n = n1 / np.linalg.norm(n1)
print('雑音n')
print(n)

x = []
x_added = []
x.append(selected_data * n)
#x_added.append(selected_data + n)
#x.append(x_added[0] / np.linalg.norm(x_added[0]))


#回帰計算
W = np.dot(S, S.T)
print('W')
print(W)

similarity = []
for j in range(calc_times):
    #print(x[j])
    similarity.append(np.absolute((np.dot(x[j].T, selected_data)).reshape(1) / N))
    x.append(np.sign(np.dot(W, x[j])))
    #x_added.append(np.sign(np.dot(W, x[j])))
    #x.append(x_added[j+1] / np.linalg.norm(x_added[j+1]))
    print(similarity[j])


#想起できているか(グラフ作成)
t = np.arange(0, calc_times, 1)
plt.plot(t, similarity)
plt.xlabel('t')
plt.ylabel('similarity')
plt.title('asociatron')
plt.xlim(0, calc_times)
plt.ylim(0, 1.1)
plt.show()
