import matplotlib.pyplot as plt
import numpy as np

global N, mag_path
#temps = np.arange(1.,4.2,.2)


def plt_mags(temps):
	fig, axes = plt.subplots(4,4)
	fig.subplots_adjust(left=.05, bottom=.05, right=.95, top=.9, wspace=.3, hspace=.5)
	axs = axes.flatten()

	for idx, temp in enumerate(temps):
		ax = axs[idx]
		filename = '{}/T{:.2f}_N{}_mag.csv'.format(mag_path,temp,N)
		mag = np.loadtxt(filename, delimiter = ',')
		ax.plot(mag/N**2)
		ax.set_title('T = {:.2f}'.format(temp))
		ax.set_ylim(-1.1,1.1)
		
	fig.suptitle("Average Magnetization/Spin vs Time Step")
	plt.show()
	return
	
def plt_set_globals(N_in, mag_path_in):
	global N, mag_path
	
	N = N_in
	mag_path = mag_path_in
	return
