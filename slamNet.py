import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from scipy.stats import multivariate_normal


from coordconv import CoordConv2d as CoordConv

class TransitionModel(nn.Module):
    def __init__(self, inputShape):
        super(TransitionModel, self).__init__()
        channels, height, width = inputShape
        self.front = nn.Sequential(
            CoordConv(channels*3, 32, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([32, height, width]),
            nn.ReLU(),
        )   
        self.convs = nn.ModuleList([
            CoordConv(32, 8, kernel_size=5, stride=1, dilation=4, padding=8),
            CoordConv(32, 8, kernel_size=5, stride=1, dilation=2, padding=4),
            CoordConv(32, 16, kernel_size=5, stride=1, dilation=1, padding=2),
            CoordConv(32, 32, kernel_size=3, stride=1, dilation=1, padding=1),
        ])
        self.body_first = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LayerNorm([64, height//2-1, width//2-1]),
            nn.ReLU(),
        )
        self.body_second = nn.Sequential(
            CoordConv(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            CoordConv(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.body_third = nn.Sequential(
            CoordConv(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            CoordConv(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.body_fourth = nn.Sequential(
            CoordConv(64, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            CoordConv(64, 16, kernel_size=4, stride=2)
        )

    def forward(self, observation, observationPrev):
        diffObservation = observation - observationPrev
        concatObservations = torch.cat((observation, observationPrev, diffObservation), dim=1)

        x = self.front(concatObservations)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)

        xi = self.body_first(x)
        xi += self.body_second(xi)
        xi += self.body_third(xi)
        x = self.body_fourth(xi)

        x = x.view(x.size(0), -1)
        
        return x
    
class GMModel(nn.Module):
    def __init__(self, num_features, k):
        super(GMModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )
        self.dense_layer = nn.ModuleList([
            nn.Linear(128, k),
            nn.Linear(128, k),
            nn.Linear(128, k),
        ])
    
    def forward(self, x):
        x = self.model(x)
        mu = self.dense_layer[0](x)
        sigma = self.dense_layer[1](x)
        logvar = self.dense_layer[2](x)
        return {'mu': mu, 'sigma': sigma, 'logvar': logvar}


class MappingModel(nn.Module):

    def __init__(self, N_ch):
        super(MappingModel, self).__init__()
        #channels, height, width = self.perspective_shape()
        channels, height, width = 1, 80, 80
        self.convs = nn.ModuleList([
            CoordConv(channels, 8, kernel_size=5, stride=1, dilation=4, padding=8),
            CoordConv(channels, 8, kernel_size=5, stride=1, dilation=2, padding=4),
            CoordConv(channels, 16, kernel_size=5, stride=1, dilation=1, padding=2),
            CoordConv(channels, 32, kernel_size=3, stride=1, dilation=1, padding=1),
        ])
        self.body_first = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([64, height//2, width//2]),
            nn.ReLU(),
        )
        self.body_second = nn.Sequential(
            CoordConv(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([32, height//2, width//2]),
            nn.ReLU(),
            CoordConv(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, height//2, width//2]),
            nn.ReLU(),
        )
        self.body_third = nn.Sequential(
            CoordConv(64, N_ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, observation):
        x = self.perspective_transform(observation)
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        xi = self.body_first(x) 
        xi += self.body_second(xi)
        x = self.body_third(xi)
        return x
    
    # TODO: Check if the wrap-perspective is working properly
    @staticmethod
    def perspective_transform(observation):
        observation_np = observation.to('cpu').numpy()
        all_images = []
        print("The shape of the observation is: ", observation_np.shape)
        for image in observation_np:
            print("The shape of the observation is: ", image.shape)
            # Move the 0th dimension to the end
            #image = np.moveaxis(image, 0, -1)
            # reshape the (c, h, w) to (w, h, c)
            image = np.reshape(image, (image.shape[2], image.shape[1], image.shape[0]))

            print("The shape of the observation is: ", image.shape)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.resize(gray_image, (80, 80))
            gray_image = gray_image.reshape(80, 80, 1)

            focal_length = 500 # in pixels
            center = (gray_image.shape[1]/2, gray_image.shape[0]/2)
            camera_matrix = np.array([[focal_length, 0, center[0]],
                                        [0, focal_length, center[1]],
                                        [0, 0, 1]], dtype = "double")

            pitch = np.radians(30)
            yaw = np.radians(45)

            rotation_vector = np.array([pitch, yaw, 0])
            rotation_matrix = cv2.Rodrigues(rotation_vector)
            translation_vector = np.array([0, 0, 10])  # in meters
            extrinsic_matrix = np.hstack((rotation_matrix[0], translation_vector.reshape(3, 1)))

            # Projection matrix
            projection_matrix = np.dot(camera_matrix, extrinsic_matrix)

            # Perspective transform
            warped_image = cv2.warpPerspective(gray_image, projection_matrix, (gray_image.shape[1], gray_image.shape[0]))
            warped_image = warped_image.reshape(80, 80, 1)
            warped_image = np.moveaxis(warped_image, -1, 0)
            all_images.append(warped_image.tolist())
        
        return torch.tensor(all_images).to('cuda')
            
    
    # @staticmethod
    # def perspective_transform(x):
    #     bs, c, h, w = x.shape
    #     c, h, w = MappingModel.perspective_shape()
    #     dummy = torch.zeros([bs, c, h, w], dtype=torch.float32, device=x.device)
    #     return dummy

class ObservationModel(nn.Module):

    def __init__(self):
        super(ObservationModel, self).__init__()
        self.body_first = nn.Sequential(
            CoordConv(2, 64, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LayerNorm(),
            nn.ReLU(),
            CoordConv(64, 32, kernel_size=3, stride=1),
        )
        self.body_second = nn.ModuleList([
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AvgPool2d(kernel_size=3, stride=2),
        ])
        self.body_third = nn.Sequential(
            nn.LayerNorm(),
            nn.ReLU(),
        )
        self.body_fourth = nn.ModuleList([
            nn.MaxPool2d(kernel_size=5, stride=5),
            nn.AvgPool2d(kernel_size=5, stride=5),
        ])
        self.body_fifth = nn.Sequential(
            nn.Linear(32*16*16, 1)
        )

    def forward(self, present_map, map_stored, particle_present, particles_stored):
        combined = self.transform(map_stored, particle_present, particles_stored)
        x = torch.cat([present_map, combined], dim=1)
        x = self.body_first(x)
        x = torch.cat([pool(x) for pool in self.body_second], dim=1)
        x = self.body_third(x)
        x = torch.cat([pool(x) for pool in self.body_fourth], dim=1)
        x = x.view(x.size(0), -1)
        x = self.body_fifth(x)
        return x
    
    @staticmethod
    def transform(map_stored, particle_present, particles_stored):
        pass



class SlamNet(nn.Module):
    def __init__(self, inputShape, K):
        super(SlamNet, self).__init__()
        # NOTE: The current implementation assumes that the states are always kept with the weight
        self.states = []
        self.states.append([[0, 0, 0] for i in range(K)])
        self.weights = []
        self.weights.append([1 for i in range(K)])
        self.K = K
        self.trajectory_estimate = [[0, 0, 0]]

        assert(len(inputShape) == 3)
        self.mapping = MappingModel(N_ch=16)
        self.visualTorso = TransitionModel(inputShape)
        if inputShape[0] == 3:
            numFeatures = 2592
        else:
        #numFeatures = 0 # need to figure out
        #self.gmmHead = GMModel(numFeatures, 3)
            self.gemHeads = nn.ModuleList([
                GMModel(numFeatures, 3),
                GMModel(numFeatures, 3),
                GMModel(numFeatures, 3),
            ])

    def calculateNewState(self, x, y, yaw):
        # x, y, yaw are all given as dictionaries

        # Choose the gaussian distribution for x, y, yaw
        x_prob = x['logvar'].exp()
        y_prob = y['logvar'].exp()
        yaw_prob = yaw['logvar'].exp()

        # Get the indices based on the probability
        x_index = torch.multinomial(x_prob, self.K, replacement=True)
        y_index = torch.multinomial(y_prob, self.K,  replacement=True)
        yaw_index = torch.multinomial(yaw_prob, self.K, replacement=True)

        assert x_index.shape == y_index.shape == yaw_index.shape == torch.Size([self.K]), "The indices are not the same size"
        # Get the new state
        new_states = []
        new_weights = []
        for i in range(self.K):
            new_x = self.states[-1][i][0] + np.random.multivariate_normal(x['mu'][x_index[i]], x['sigma'][x_index[i]])
            new_y = self.states[-1][i][1] + np.random.multivariate_normal(y['mu'][y_index[i]], y['sigma'][y_index[i]])
            new_yaw = self.states[-1][i][2] + np.random.multivariate_normal(yaw['mu'][yaw_index[i]], yaw['sigma'][yaw_index[i]])
            new_states.append([new_x, new_y, new_yaw])
        
            # Calculate the new weights
            new_weight_x = self.weights[-1][i] * multivariate_normal.pdf(new_x, x['mu'][x_index[i]], x['sigma'][x_index[i]])
            new_weight_y = self.weights[-1][i] * multivariate_normal.pdf(new_y, y['mu'][y_index[i]], y['sigma'][y_index[i]])
            new_weight_yaw = self.weights[-1][i] * multivariate_normal.pdf(new_yaw, yaw['mu'][yaw_index[i]], yaw['sigma'][yaw_index[i]])
            new_weights.append(new_weight_x * new_weight_y * new_weight_yaw)
        
        # Normalize the weights
        new_weights = new_weights / np.sum(new_weights)
        self.states.append(new_states)
        self.weights.append(new_weights)
        return new_states, new_weights

    def calc_average_trajectory(self, new_states, new_weights):
        # Calculate the average trajectory
        pose_estimate = [0, 0, 0]
        for i in range(self.K):
            pose_estimate[0] += new_states[i][0] * new_weights[i]
            pose_estimate[1] += new_states[i][1] * new_weights[i]
            pose_estimate[2] += new_states[i][2] * new_weights[i]
        self.trajectory_estimate.append(pose_estimate)
        return pose_estimate

    def forward(self, observation, observationPrev):
        map_t = self.mapping(observation)
        featureVisual = self.visualTorso(observation, observationPrev)
        gmm = self.gmmHead(featureVisual)
        #return {'map': map_t, 'gmm': gmm}
        x = self.gemHeads[0](featureVisual)
        y = self.gemHeads[1](featureVisual)
        yaw = self.gemHeads[2](featureVisual)
    
        # Using x, y, yaw to calculate the new state
        new_states, new_weights = self.calculateNewState(x, y, yaw)

        # Calculate the resultant pose estimate
        pose_estimate = self.calc_average_trajectory(new_states, new_weights)
        print("The estiamted current pose is: ", pose_estimate)

        return {'x': x, 'y': y, 'yaw': yaw}

