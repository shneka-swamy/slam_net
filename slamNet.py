import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from torch.distributions import MultivariateNormal


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
            
    
    @staticmethod
    def perspective_transform_dummy(x):
        bs, c, h, w = x.shape
        c, h, w = (1, 80, 80)
        dummy = torch.zeros([bs, c, h, w], dtype=torch.float32, device=x.device)
        return dummy

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


# NOTE: In general the model is in testing mode
"""
Apart from that the model can be in one of the following modes:
1. Training mode
2. Pretraining mode -- Transition model
3. Pretraining mode -- Observation and mapping model
"""
class SlamNet(nn.Module):
    # TODO: This function needs to be updated with the training or testing mode parameter
    def __init__(self, inputShape, K, is_training=False, is_pretrain_trans=False, is_pretrain_obs=False):
        super(SlamNet, self).__init__()
        # NOTE: The current implementation assumes that the states are always kept with the weight
        self.bs = inputShape[0]
        self.states = torch.zeros([self.bs, K, 3], dtype=torch.float32, device='cuda')
        self.weights = torch.ones([self.bs, K], dtype=torch.float32, device='cuda')
        self.lastStates = [0,0,0]
        self.lastWeights = 1
        self.K = K
        self.trajectory_estimate = [[0, 0, 0]]

        self.is_training = is_training
        self.is_pretrain_trans = is_pretrain_trans
        self.is_pretrain_obs = is_pretrain_obs

        assert(len(inputShape) == 4)
        #self.mapping = MappingModel(N_ch=16)
        self.visualTorso = TransitionModel(inputShape[1:])
        if inputShape[1] == 3:
            numFeatures = 2592
        else:
            numFeatures = 2592
        self.gemHeads = nn.ModuleList([
            GMModel(numFeatures, 3),
            GMModel(numFeatures, 3),
            GMModel(numFeatures, 3),
        ])

    # TODO: This function needs more information -- needs to get the ground truth
    # TODO: Please use ground truth parameter for this purpose
    def forward(self, observation, observationPrev):
        if self.is_training or self.is_pretrain_obs:
            map_t = self.mapping(observation)
        
        if self.is_training or self.is_pretrain_trans:
            featureVisual = self.visualTorso(observation, observationPrev)
            x = self.gemHeads[0](featureVisual)
            y = self.gemHeads[1](featureVisual)
            yaw = self.gemHeads[2](featureVisual)

            if self.is_pretrain_trans:
                # Using x, y, yaw to calculate the new state
                new_states, new_weights = self.calculateNewState(x, y, yaw)
                print(new_states.shape, new_weights.shape)
        
        #new_states, new_weights = self.resample(new_states, new_weights)
        #print(new_states.shape, new_weights.shape)

        # Calculate the resultant pose estimate
        pose_estimate = self.calc_average_trajectory(new_states, new_weights)

        # Calculate the loss between the estimated pose and the ground truth pose
        ground_truth = torch.randn([self.bs, 3], dtype=torch.float32)
        loss = self.huber_loss(pose_estimate, ground_truth)

        # TODO: Can return loss instead -- whatever is required for backward pass
        return {'x': x, 'y': y, 'yaw': yaw}


    # find the huber loss between the estimated pose and the ground truth pose
    # The value of delta can be altered
    def huber_loss (self, pose_estimated, actual_pose, delta = 0.1):
        residual = torch.abs(pose_estimated - actual_pose)
        is_small_res = residual < delta
        return torch.where(is_small_res, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))        

    # Resample the particles based on the weights
    # NOTE: Paper does not mention if the resampling is hard or soft and hence we use soft to avaoid zero gradient
    # NOTE: This function is a PyTorch version of the PFNet implementation
    def resample(self, particle_states, particle_weights, alpha=torch.tensor([0])):
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


    def calculateNewState(self, x, y, yaw):
        # x, y, yaw are all given as dictionaries
        # x, y, yaw cannot be joined as all of them have different GMM
        mean = torch.stack([x['mu'], y['mu'], yaw['mu']])

        # Make sigma a diagonal matrix with batch size as the first dimension
        x['cov_diag'] = torch.diag_embed(x['sigma']**2)  
        x['covariance'] = torch.bmm(x['cov_diag'], x['cov_diag'].transpose(1, 2))
        y['cov_diag'] = torch.diag_embed(y['sigma']**2)
        y['covariance'] = torch.bmm(y['cov_diag'], y['cov_diag'].transpose(1, 2))
        yaw['cov_diag'] = torch.diag_embed(yaw['sigma']**2)
        yaw['covariance'] = torch.bmm(yaw['cov_diag'], yaw['cov_diag'].transpose(1, 2))

        covariance = torch.stack([x['covariance'], y['covariance'], yaw['covariance']])
        prob_stack = torch.stack([x['logvar'], y['logvar'], yaw['logvar']])
        prob_stack = torch.exp(prob_stack)

        new_states = torch.zeros(self.bs, self.K, 3)
        new_weights = torch.zeros(self.bs, self.K)

        for i in range(3):                                            
            for j in range(self.bs):
                component_samples = torch.multinomial(prob_stack[i][j], self.K, replacement=True)
                for k, component in enumerate(component_samples):
                    mean_component = mean[i][j]
                    covariance_component = covariance[i][j]
                    mvn = MultivariateNormal(mean_component, covariance_component) 
                    chosen_sample = mvn.sample()
                    new_states[j][k][i] = self.states[j][k][i] + chosen_sample[component]
                    new_weights[j][k] = self.weights[j][k] * mvn.log_prob(chosen_sample)
        self.states = new_states
        self.weights = new_weights      
        return new_states, new_weights
    

    def calc_average_trajectory(self, new_states, new_weights):
        # Calculate the average trajectory
        pose_estimate = torch.zeros(self.bs, 3)
        for i in range(self.K):
            pose_estimate[:, 0] += new_states[:, i, 0] * new_weights[:, i]
            pose_estimate[:, 1] += new_states[:, i, 1] * new_weights[:, i]
            pose_estimate[:, 2] += new_states[:, i, 2] * new_weights[:, i]
        self.trajectory_estimate.append(pose_estimate)
        return pose_estimate

