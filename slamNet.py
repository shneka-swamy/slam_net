import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily

from coordconv import CoordConv2d as CoordConv

class TransitionModel(nn.Module):
    def __init__(self, inputShape, use_cuda):
        super(TransitionModel, self).__init__()
        channels, height, width = inputShape
        self.front = nn.Sequential(
            CoordConv(channels*3, 32, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.LayerNorm([32, height, width]),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList([
            CoordConv(32, 8, kernel_size=5, stride=1, dilation=4, padding=8, use_cuda=use_cuda),
            CoordConv(32, 8, kernel_size=5, stride=1, dilation=2, padding=4, use_cuda=use_cuda),
            CoordConv(32, 16, kernel_size=5, stride=1, dilation=1, padding=2, use_cuda=use_cuda),
            CoordConv(32, 32, kernel_size=3, stride=1, dilation=1, padding=1, use_cuda=use_cuda),
        ])
        self.body_first = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LayerNorm([64, height//2-1, width//2-1]),
            nn.ReLU(),
        )
        self.body_second = nn.Sequential(
            CoordConv(64, 128, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.ReLU(),
            CoordConv(128, 64, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.ReLU(),
        )
        self.body_third = nn.Sequential(
            CoordConv(64, 128, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.ReLU(),
            CoordConv(128, 64, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.ReLU(),
        )
        self.body_fourth = nn.Sequential(
            CoordConv(64, 64, kernel_size=4, stride=2, use_cuda=use_cuda),
            nn.ReLU(),
            CoordConv(64, 16, kernel_size=4, stride=2, use_cuda=use_cuda)
        )

    def forward(self, observation, observationPrev):
        diffObservation = observation - observationPrev

        concatObservations = torch.cat((observation, observationPrev, diffObservation), dim=1)

        x_front = self.front(concatObservations)
        x_conv = []
        for conv in self.convs:
            x_new = conv(x_front)
            x_conv.append(x_new)

        x_cat = torch.cat(x_conv, dim=1)
        #x_conv = torch.cat([conv(x) for conv in self.convs], dim=1)

        xi1 = self.body_first(x_cat)
        xi2 = xi1 + self.body_second(xi1)
        xi3 = xi2 + self.body_third(xi2)
        x_fourth = self.body_fourth(xi3)

        x_final = x_fourth.view(x_fourth.size(0), -1)

        return x_final

class GMModel(nn.Module):
    def __init__(self, num_features, k):
        super(GMModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, k)
        self.sigma = nn.Linear(128, k)
        self.logvar = nn.Linear(128, k)
        

    def forward(self, x):
        xn = self.model(x)
        mu = self.mu(xn)
        sigma = self.sigma(xn)
        logvar = self.logvar(xn)
        return mu, sigma, logvar
       #return {'mu': mu, 'sigma': sigma, 'logvar': logvar}

class MappingModel(nn.Module):

    def __init__(self, N_ch, use_cuda):
        super(MappingModel, self).__init__()
        #channels, height, width = self.perspective_shape()
        channels, height, width = 1, 80, 80
        self.convs = nn.ModuleList([
            CoordConv(channels, 8, kernel_size=5, stride=1, dilation=4, padding=8, use_cuda=use_cuda),
            CoordConv(channels, 8, kernel_size=5, stride=1, dilation=2, padding=4, use_cuda=use_cuda),
            CoordConv(channels, 16, kernel_size=5, stride=1, dilation=1, padding=2, use_cuda=use_cuda),
            CoordConv(channels, 32, kernel_size=3, stride=1, dilation=1, padding=1, use_cuda=use_cuda),
        ])
        self.body_first = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([64, height//2, width//2]),
            nn.ReLU(),
        )
        self.body_second = nn.Sequential(
            CoordConv(64, 32, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.LayerNorm([32, height//2, width//2]),
            nn.ReLU(),
            CoordConv(32, 64, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda),
            nn.LayerNorm([64, height//2, width//2]),
            nn.ReLU(),
        )
        self.body_third = nn.Sequential(
            CoordConv(64, N_ch, kernel_size=3, stride=1, padding=1, use_cuda=use_cuda)
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

class ObservationModel(nn.Module):

    def __init__(self, use_cuda):
        super(ObservationModel, self).__init__()
        self.body_first = nn.Sequential(
            CoordConv(2, 64, kernel_size=5, stride=1, use_cuda=use_cuda),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LayerNorm(),
            nn.ReLU(),
            CoordConv(64, 32, kernel_size=3, stride=1, use_cuda=use_cuda),
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


# NOTE: In general the model is in testing mode
"""
Apart from that the model can be in one of the following modes:
1. Training mode
2. Pretraining mode -- Transition model
3. Pretraining mode -- Observation and mapping model
"""
class SlamNet(nn.Module):
    # TODO: This function needs to be updated with the training or testing mode parameter
    def __init__(self, inputShape, K, is_training=False, is_pretrain_trans=False, is_pretrain_obs=False,
                 use_cuda=True):
        super(SlamNet, self).__init__()
        # NOTE: The current implementation assumes that the states are always kept with the weight
        self.bs = inputShape[0]
        self.lastStates = [0,0,0]
        self.lastWeights = 1
        self.K = K
        self.trajectory_estimate = [[0, 0, 0]]

        self.is_training = is_training
        self.is_pretrain_trans = is_pretrain_trans
        self.is_pretrain_obs = is_pretrain_obs

        assert(len(inputShape) == 4)
        self.mapping = MappingModel(N_ch=16, use_cuda=use_cuda)
        self.visualTorso = TransitionModel(inputShape[1:], use_cuda=use_cuda)
        if inputShape[1] == 3:
            numFeatures = 2592
        else:
            numFeatures = 2592
        self.gmmX = GMModel(numFeatures, 3)
        self.gmmY = GMModel(numFeatures, 3)
        self.gmmYaw = GMModel(numFeatures, 3)

    # TODO: This function needs more information -- needs to get the ground truth
    # TODO: Please use ground truth parameter for this purpose
    def forward(self, observation, observationPrev):
        if self.is_training or self.is_pretrain_obs:
            map_t = self.mapping(observation)

        if self.is_training or self.is_pretrain_trans:
            featureVisual = self.visualTorso(observation, observationPrev)
            x = self.gmmX(featureVisual)
            y = self.gmmY(featureVisual)
            yaw = self.gmmYaw(featureVisual)

        # Testing till x, y, yaw
        #return x, y, yaw


            if self.is_pretrain_trans:
                    # Using x, y, yaw to calculate the new state
                    # Testing if only x can work
                    new_states, new_weights = self.tryNewState(x, y , yaw)
                    #new_states, new_weights = self.calculateNewStateDummy(x) #, y, yaw)
                    #print(new_states.shape, new_weights.shape)

        #new_states, new_weights = self.resample(new_states, new_weights)
        #print(new_states.shape, new_weights.shape)

        # Calculate the resultant pose estimate
        pose_estimate = self.calc_average_trajectory(new_states, new_weights)

        # TODO: Can return loss instead -- whatever is required for backward pass
        return pose_estimate


    # find the huber loss between the estimated pose and the ground truth pose
    # The value of delta can be altered
    @staticmethod
    def huber_loss(pose_estimated, actual_pose, delta = 0.1):
        residual = torch.abs(pose_estimated - actual_pose)
        is_small_res = residual < delta
        return torch.where(is_small_res, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))

    # Resample the particles based on the weights
    # NOTE: Paper does not mention if the resampling is hard or soft and hence we use soft to avaoid zero gradient
    # NOTE: This function is a PyTorch version of the PFNet implementation
    @staticmethod
    def resample(particle_states, particle_weights, alpha=torch.tensor([0])):
        batch_size, num_particles = particle_states.shape[:2]

        # normalize
        particle_weights = particle_weights - torch.logsumexp(particle_weights, dim=-1, keepdim=True)

        uniform_weights = torch.full((batch_size, num_particles), -torch.log(torch.tensor(num_particles)), dtype=torch.float32)

        # build sampling distribution, q(s), and update particle weights
        if alpha < 1.0:
            # soft resampling
            q_weights = torch.stack([particle_weights + torch.log(alpha), uniform_weights + torch.log(1.0-alpha)], dim=-1)
            q_weights = torch.logsumexp(q_weights, dim=-1, keepdim=False)
            q_weights = q_weights - torch.logsumexp(q_weights, dim=-1, keepdim=True)  # normalized

            particle_weights = particle_weights - q_weights  # this is unnormalized
        else:
            # hard resampling. this will produce zero gradients
            q_weights = particle_weights
            particle_weights = uniform_weights

        # sample particle indices according to q(s)
        indices = torch.multinomial(torch.exp(q_weights), num_particles, replacement=True)  # shape: (batch_size, num_particles)

        # index into particles
        helper = torch.arange(0, batch_size*num_particles, step=num_particles, dtype=torch.int64)  # (batch, )
        indices = indices + helper.view(batch_size, 1).to(indices.device)

        particle_states = particle_states.view(batch_size * num_particles, 3)
        particle_states = particle_states.index_select(0, indices.view(-1)).view(batch_size, num_particles, 3)

        particle_weights = particle_weights.view(batch_size * num_particles)
        particle_weights = particle_weights.index_select(0, indices.view(-1)).view(batch_size, num_particles)

        return particle_states, particle_weights

    def tryNewState(self, x, y, yaw):
        x_mu, x_sigma, x_logvar = x
        y_mu, y_sigma, y_logvar = y
        yaw_mu, yaw_sigma, yaw_logvar = yaw

        mean_val = torch.cat([x_mu, y_mu, yaw_mu], dim=0)
        std_values = torch.cat([x_sigma , y_sigma, yaw_sigma], dim=0)
        # Mean wise multiplication of logvar
        logvar = []
        for i in range(3):
            new_value = x_logvar[:, i] * y_logvar[:, i] * yaw_logvar[:, i]
            logvar.append(new_value)
        logvar = torch.cat(logvar, dim=0)

        std_values = torch.nn.functional.softplus(std_values)
        logvar = torch.nn.functional.softplus(logvar)

        distributions = Independent(Normal(mean_val, std_values), 1)
        mixture_dist = Categorical(logvar)

        gmm_dist = MixtureSameFamily(mixture_dist, distributions) 

        samples = gmm_dist.sample(torch.Size([self.bs, self.K]))

        log_probs = gmm_dist.log_prob(samples)
        weights = torch.exp(log_probs - torch.max(log_probs))
        weights = weights / torch.sum(weights)

        return samples, weights

    def calc_average_trajectory(self, new_states, new_weights):
        # Calculate the average trajectory
        pose_estimate = torch.zeros(self.bs, 3, device=new_states.device)
        for i in range(self.K):
            pose_estimate[:, 0] = pose_estimate[:,0] + new_states[:, i, 0] * new_weights[:, i]
            pose_estimate[:, 1] = pose_estimate[:,1] + new_states[:, i, 1] * new_weights[:, i]
            pose_estimate[:, 2] = pose_estimate[:,2] + new_states[:, i, 2] * new_weights[:, i]
        #self.trajectory_estimate.append(pose_estimate)
        return pose_estimate

