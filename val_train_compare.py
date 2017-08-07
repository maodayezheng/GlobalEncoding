import numpy as np
import matplotlib.pyplot as plt

folder = "code_outputs/"
name = "2017_07_21_10_11_54/"
loss = "rnn_loss.npy"
rnn_loss_2 = np.load(folder + name + loss)
rnn_loss_2 = np.mean(rnn_loss_2[:-10].reshape((600, 100)), axis=-1)
plt.plot(rnn_loss_2, label='Auto encoder', zorder=1)
plt.xlabel('Negative log likelihood on rnnlm (Averaged over 100 iterations)')
plt.ylabel('L(X))')
plt.legend(loc=1)
plt.tight_layout()
plt.savefig('ae_loss' + '.png')

"""
total_elbos =0
for i in range(len(training)):
    total_elbos += training[i]
    if (i+1) % 200 is 0:
        aver_elbos.append(total_elbos/200)
        total_elbos = 0

plt.plot(aver_elbos, label='train', zorder=2)
plt.plot(validation, label='test', zorder=1)

plt.xlabel('Training elbo (Average over 200 iters) VS Testing elbo (Calculate every 200 time step)')
plt.ylabel('L(X))')
plt.legend(loc=4)
plt.tight_layout()

plt.savefig('code_outputs/DecoderAddress-10k' + '.png')

plt.clf()
"""