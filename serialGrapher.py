import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#credit to B.K.

for a in range(1, 82):
     data = pd.read_csv('C:/Users/alper/Desktop/new grapher/{:d}.csv'.format(a)).reset_index()
     data = data[['index', 'x', 'y']]
     data = data.rename(columns = {'index': 'frame', 'x': 'xCoord', 'y': 'yCoord'})
     data['frame'] = data['frame'] + 1
     
     with plt.style.context('seaborn-dark'):
          plt.rcParams['figure.figsize'] = [40, 10]
          plt.rcParams['lines.linewidth'] = 5

          data = data[0:600]
          frame = data['frame']
          x_cord = data['xCoord']
          plt.plot(frame, x_cord, label = "Movement")
     plt.savefig('{:d}_haar.png'.format(a))
     plt.close()
