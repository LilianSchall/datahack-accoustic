import matplotlib.pyplot as plt
import matplotlib
from pylab import rcParams
import numpy as np
from sklearn.preprocessing import minmax_scale

class RoomSetup(object):
    def __init__(self,
                speaker_xyz,
                mic_xyzs,
                x_min: float,
                x_max: float,
                y_min: float,
                y_max: float,
                walls=None):

        self.speaker_xyz = speaker_xyz
        self.mic_xyzs = mic_xyzs
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.walls = walls
        self.n_mics = mic_xyzs.shape[0]

    def plot_room(self, centroid, levels=None, legend=True, camera_coords=True, vmax=0.3, vmin=0.1945841):

        
        c = centroid

        #Scatter Plot
        rcParams['figure.figsize'] = 8,8
        plt.scatter(self.speaker_xyz[0], self.speaker_xyz[1], label='Speaker', color='green')
        plt.scatter(self.mic_xyzs[:,0], self.mic_xyzs[:,1], label = 'Mics', color='orange')

        if levels is None:
            plt.scatter(c[:,0], c[:,1], label = 'Centroids', c ='green', s=4)
        else:
            plt.scatter(c[:,0], c[:,1], label = 'Human Locations', c=levels, cmap="bwr", s=9)

        if self.walls is not None:
            plt.plot(self.walls[:,0], self.walls[:,1] , marker = 'o', color='black', label = 'Walls')

        #plt.scatter(0, 0, label = 'Origin', c ='black')
        plt.xlim([self.x_min, self.x_max])
        plt.ylim([self.y_min, self.y_max])
        plt.axis('equal')

        if legend:
            plt.legend()

    def plot_prediction(self, pred, verita_tera, legend=True, camera_coords=True, vmax=0.3, vmin=0.1945841):
        #Scatter Plot
        rcParams['figure.figsize'] = 8,8
        plt.scatter(self.speaker_xyz[0], self.speaker_xyz[1], label='Speaker', color='green')
        plt.scatter(self.mic_xyzs[:,0], self.mic_xyzs[:,1], label = 'Mics', color='orange')

        for i in range(pred.shape[0]):
            plt.plot((pred[i,0], verita_tera[i,0]), (pred[i,1], verita_tera[i,1]), "r-", linewidth=1)

        plt.scatter(verita_tera[:,0], verita_tera[:,1], label = 'Verita Tera', c ='green', s=4)
        plt.scatter(pred[:,0], pred[:,1], label = 'Prediction', c ='blue', s=4)

        if self.walls is not None:
            plt.plot(self.walls[:,0], self.walls[:,1] , marker = 'o', color='black', label = 'Walls')

        plt.xlim([self.x_min, self.x_max])
        plt.ylim([self.y_min, self.y_max])
        plt.axis('equal')

        if legend:
            plt.legend()

    def plot_heatmap(self, pred, verita_tera, err, legend=True, camera_coords=True, vmax=0.3, vmin=0.1945841):
        #Scatter Plot
        rcParams['figure.figsize'] = 8,8
        plt.scatter(self.speaker_xyz[0], self.speaker_xyz[1], label='Speaker', color='green')
        plt.scatter(self.mic_xyzs[:,0], self.mic_xyzs[:,1], label = 'Mics', color='orange')

        dist = np.linalg.norm(pred - verita_tera, axis=1)
        max_dist, min_dist = np.max(dist), np.min(dist)
        print("max dist =", max_dist)
        print("min dist =", min_dist)
        norm_dist = minmax_scale(dist)
        colors = []
        for i in range(dist.shape[0]):
            #colors.append((norm_dist[i]if norm_dist[i] > 0.33 else 0, norm_dist[i] if norm_dist[i] > 0.66 else 0, norm_dist[i], norm_dist[i]))
            colors.append((norm_dist[i], 1.0 - norm_dist[i], 0, norm_dist[i]))

        norm=plt.Normalize(min_dist, max_dist)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "red"])
        plt.scatter(verita_tera[:,0], verita_tera[:,1], c=colors, s=(dist / 2))

        img = plt.imshow(np.array([[0,1]]), norm=norm, cmap=cmap)
        img.set_visible(False)
        plt.colorbar(orientation="vertical", label="Distance")

        #for i in range(pred.shape[0]):
        #    plt.scatter(verita_tera[i,0], verita_tera[i,1], c ='red', s=(dist[i]/10))

        plt.scatter(verita_tera[:,0], verita_tera[:,1], label = 'Verita Tera', c ='green', s=4)
        #plt.pcolormesh(verita_tera[:,0], verita_tera[:,1], dist)

        if self.walls is not None:
            plt.plot(self.walls[:,0], self.walls[:,1] , marker = 'o', color='black', label = 'Walls')

        
        plt.xlim([self.x_min, self.x_max])
        plt.ylim([self.y_min, self.y_max])
        plt.axis('equal')
        
        # ax = plt.gca()
        # ax.set_facecolor((0, 0, 0))

        if legend:
            plt.legend()

