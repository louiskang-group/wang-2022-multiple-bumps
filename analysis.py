import numpy as np
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from functools import partial
from scipy.ndimage import gaussian_filter1d

import utils


font = {
    "family": "serif",
    "color": "black",
    "weight": "bold",
    "size": 15,
}


class DiffusionAnalysis:
    def __init__(
        self, device, positions, nNeurons, dt=0.5/1000, uMaxFac=0.5, diffusionCutoff=0
    ):

        """
        Performs diffusion analysis to determine diffusion constants and integration velocities
        of the bumps.
        
        Inputs
        ------
        device : cuda device name
        positions : array over replicate simulations. Dim: (number of replicates, simulation duration, number of bumps) 
        nNeurons : number of neurons in each population
        dt : timestep (in seconds)
        uMaxFac : time offset maximum as proportion of simulation time
        diffusionCutoff : remove short timescales below this index during fit; not used in manuscript
        
        """
        # ts are simulation times
        tMax = positions.shape[1]
        ts = dt * torch.arange(tMax, device=device)
        # us are time offsets
        uMax = int(round(uMaxFac * tMax))
        us = dt * torch.arange(1, uMax, device=device)
        usFit = us.unsqueeze(-1).double()
        # When we remove short timescales, we also allow for a y-intercept during,
        # the fit, which is achieved by stacking on a list of 1's
        if diffusionCutoff != 0:
            usStack = usFit[diffusionCutoff:]
            usStack = torch.hstack((usStack, torch.ones_like(usStack)))

        # Find position increments and unwrap jumps through the periodic boundary
        increments = torch.diff(torch.tensor(positions, device=device), dim=1)
        increments = utils.mod_offset(increments, nNeurons, -nNeurons / 2)
        
        # Calculate the unwrapped displacements and add a 0 to the beginning of
        # each time series
        displacements = torch.cumsum(increments, dim=1)
        displacements = torch.hstack(
            (torch.zeros_like(displacements[:, 0:1]), displacements)
        ) 

        # Compute mean differences between all trajectory segments separated by u timesteps
        integrations = torch.stack(
            [
                (displacements[:, u:] - displacements[:, :-u]).mean((0, 1))
                for u in range(1, uMax)
            ]
        )
        
        # Fit integrations with a line through the origin to get velocity
        integrationVels = torch.lstsq(integrations, usFit)[0][0]

        # Subtract the mean bump motion across replicate simulations
        residuals = displacements - displacements.mean(0, keepdim=True)
        
        # Position changes over u timesteps, squared and averaged over times and replicates
        diffusions = torch.stack(
            [
                ((residuals[:, u:] - residuals[:, :-u]) ** 2).mean((0, 1))
                for u in range(1, uMax)
            ]
        )
        
        # Fit diffusions with a line through the origin to get velocity
        if diffusionCutoff == 0:
            diffusionConsts = torch.lstsq(diffusions, usFit)[0][0] / 2
        else:
            diffusionConsts = (
                torch.lstsq(diffusions[diffusionCutoff:], usStack)[0][0] / 2
            )

        self.integrations = integrations.cpu().numpy()
        self.diffusions = diffusions.cpu().numpy()
        self.integrationVels = integrationVels.cpu().numpy()
        self.diffusionConsts = diffusionConsts.cpu().numpy()
        self.us = us.cpu().numpy()

    def plot_integration(self):
        for integration, integrationVel in zip(
            self.integrations.T, self.integrationVels
        ):
            plt.plot(self.us, integration)
            plt.plot(self.us[(0, -1),], integrationVel * self.us[(0, -1),], color="0.5")
        plt.show()

    def plot_diffusion(self):
        for diffusion, diffusionConst in zip(self.diffusions.T, self.diffusionConsts):
            plt.plot(self.us, diffusion)
            plt.plot(
                self.us[(0, -1),], 2 * diffusionConst * self.us[(0, -1),], color="0.5"
            )
        plt.show()


def bootstrap_diffusion(
    device, positionsAll, nNeurons, dt=0.5/1000, diffusionCutoff=0
):
    
    # Take bootstrap sample of replicates and perform diffusion analysis
    nReps = positionsAll.shape[0]
    bootstrap = positionsAll[np.random.choice(np.arange(nReps), nReps)]

    a = DiffusionAnalysis(
        device, bootstrap, nNeurons, dt=dt, diffusionCutoff=diffusionCutoff
    )

    return a.integrationVels, a.diffusionConsts


class IntegrationAnalysis:
    def __init__(self, device, positions, nNeurons, dt=0.5/1000, uMaxFac=0.5):
        
        """
        Similar to DiffusionAnalysis but only calculate integration velocities, not diffusion constants.
       
        """

        tMax = positions.shape[0]
        ts = dt * torch.arange(tMax, device=device)
        
        uMax = int(round(uMaxFac * tMax))
        us = dt * torch.arange(1, uMax, device=device)
        usFit = us.unsqueeze(-1).double()

        increments = torch.diff(torch.tensor(positions, device=device), dim=0)
        increments = utils.mod_offset(increments, nNeurons, -nNeurons / 2)
        displacements = torch.cumsum(increments, dim=0)
        displacements = torch.vstack(
            (torch.zeros_like(displacements[0:1]), displacements)
        )

        integrations = torch.stack(
            [(displacements[u:] - displacements[:-u]).mean(0) for u in range(1, uMax)]
        )
        integrationVels = torch.lstsq(integrations, usFit)[0][0]

        self.integrations = integrations.cpu().numpy()
        self.integrationVels = integrationVels.cpu().numpy()
        self.us = us.cpu().numpy()

    def plot_integration(self):
        for integration, integrationVel in zip(
            self.integrations.T, self.integrationVels
        ):
            plt.plot(self.us, integration)
            plt.plot(self.us[(0, -1),], integrationVel * self.us[(0, -1),], color="0.5")
        plt.show()


class VelocityAnalysis:
    def __init__(self, simObj, tBlur=20, truncate=3.0, dt=0.5 / 1000):
        
        """
        Calculate bump velocity as a function of position
        
        Input
        -----
        simObj : GPUSim object
        tBlur : Gaussian standard deviation for smoothing bump velocities in time
        truncate : truncate Gaussian filter beyond this number of standard deviations
        dt : timestep (in seconds)
        """

        self.nNeurons = simObj.nNeurons
        self.nBumps = simObj.nBumps
        positions = simObj.positions

        # Computing velocities while unwrapping jumps around the periodic boundary
        velocities = np.diff(positions, axis=0) 
        velocities = (
            utils.mod_offset(velocities, self.nNeurons, -self.nNeurons / 2) / dt
        )

        # Smooth velocities in time with Gaussian filter. Trim ends of data where
        # smoothing is incomplete
        tTrim = int(np.ceil(truncate * tBlur) + 1)
        velocities = gaussian_filter1d(velocities, tBlur, truncate=truncate, axis=0)[
            tTrim:-tTrim
        ] 
        self.velocities = velocities
        
        neurons = positions[tTrim : -tTrim - 1].astype(int)

        # Each list in velocityTable records all velocities occurring at a integer
        # neural position
        velocityTable = [[] for _ in range(self.nNeurons)]

        for neuron, velocity in zip(
            np.concatenate(neurons), np.concatenate(velocities)
        ):
            velocityTable[neuron].append(velocity)
         
        # Record means and standard deviations at each neural position
        self.velocityTable = velocityTable
        self.velocityMeans = np.array([np.mean(velocity) for velocity in velocityTable])
        self.velocityStds = np.array([np.std(velocity) for velocity in velocityTable])

    def plot_velocity(self):
        plt.plot(self.velocityMeans + self.velocityStds, color="0.7")
        plt.plot(self.velocityMeans - self.velocityStds, color="0.7")
        plt.plot(self.velocityMeans)
        plt.show()


class SpeedVariationAnalysis:
    def __init__(self, velocityObjR, velocityObjL):
        """
        Perform speed variability and speed difference analysis
        
        Inputs 
        ------
        velocityObjR : an VelocityAnalysis object for R population
        velocityObjL : an VelocityAnalysis object for L population
        """
       
        self.nNeurons = velocityObjR.nNeurons
        self.nBumps = velocityObjR.nBumps

        self.speedsR = velocityObjR.velocityMeans 
        self.speedsL = -velocityObjL.velocityMeans  # speedsL are positive

        self.meanR = np.nanmean(self.speedsR)  # Mean speed of R population
        self.meanL = np.nanmean(self.speedsL)  # Mean speed of L population
        self.mean = np.mean((self.meanR, self.meanL))
        self.difference = (self.meanR - self.meanL) / self.mean  # Normalized difference of means

        self.variabilityR = np.nanstd(self.speedsR) / self.mean  # Speed variability for R population
        self.variabilityL = np.nanstd(self.speedsL) / self.mean  # Speed variability for L population
        self.variability = np.mean((self.variabilityR, self.variabilityL)) # Average speed variability

    def plot_speed(self):
        plt.plot(self.speedsR)
        plt.plot(self.speedsL)
        plt.show()


class InformationAnalysis:
    def __init__(self, device, activities, nCues=1, nBins=6, scale=None):
        
        """
        Perform mutual information analysis 
        
        Inputs
        ------
        device : cuda device name
        activities : firing rates concatenated across populations. Dim: (nReps, nPositions, 2*nNeurons)
        nCues : number of equally-spaced local cues
        nBins : number of activity bins
        scale : evenly distribute bins from 0 to scale
        """

        self.nCues = nCues
        self.nBins = nBins
        self.nStates = nCues * nBins  # Joint activity states
        if scale is None:  # Use 99th percentile by default
            self.scale = np.quantile(activities, 0.99)
        else:
            self.scale = scale

        self.nReps = activities.shape[0]  # Number of replicate simulations per position
        self.nPositions = activities.shape[1]  # Number of coordinate positions, assumed equally spaced
        self.nNeurons = activities.shape[2] / 2

        # Binning activities into single-neuron states
        scaledActivities = torch.tensor(activities, device=device) / self.scale
        activityStates = torch.clamp((nBins * scaledActivities).int(), max=nBins-1)

        # Now we account for joint activity states by shifting state indices to new
        # values for positions corresponding to new cues. To do so, we increment all state
        # indices by nBins every cueLength positions.
        cueLength = (int(np.ceil(self.nPositions / nCues)),)
        cueOffsets = torch.cat(
            [torch.full(cueLength, cue * nBins, device=device) for cue in range(nCues)]
        )[: self.nPositions]
        activityStates += cueOffsets.unsqueeze(0).unsqueeze(2)

        # Tallying states to determine activity probabilities p(s|u)
        activityProbs = (
            torch.stack(
                [
                    (activityStates == state).count_nonzero(0)
                    for state in range(self.nStates)
                ]
            ).permute(2, 1, 0)
            / self.nReps
        )

        # Calculating mutual information from activityProbs according to Eq. 22
        self.informations = (
            torch.sum(
                torch.nan_to_num(
                    activityProbs
                    * torch.log2(activityProbs / activityProbs.mean(1, keepdim=True))
                ),
                (1, 2),
            )
            / self.nPositions
        )

        self.informations = self.informations.cpu().numpy()

    def plot_informations(self):
        plt.hist(self.informations)
        plt.show()


def bootstrap_mean_information(device, activitiesAll, nCues=1):

    # Take bootstrap sample of replicates and perform diffusion analysis
    nReps = activitiesAll.shape[0]
    bootstrap = activitiesAll[np.random.choice(np.arange(nReps), nReps)]

    a = InformationAnalysis(device, bootstrap, nCues=nCues)

    return a.informations.mean()
