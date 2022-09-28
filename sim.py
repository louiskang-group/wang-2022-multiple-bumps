import sys
import numpy as np
import utils
from matplotlib.animation import FuncAnimation
from IPython import display
import matplotlib.pyplot as plt
import torch


class GPUSim:
    """
    A class for simulating a continuous attractor network on GPU
    
    ...
    
    Attributes
    ----------
    device (str) : name of cuda device
    nNeurons (int) : number of neurons in each L and R population (N)
    nBumps (int) : number of activity bumps in the network (M)
    dt (float) : simulation timestep
    tau (float) : time constant
    phiFunction (func) : neural transfer function
    wAttractor (tensor) : connectivity weights
    gsL (array) : synaptic inputs of left neurons
    gsR (array) : synaptic inputs of right neurons
    
    
    Methods
    -------
    evolve_network : one simulation timestep update to network 
    simulate : full simulation network
    movie : produce movie for visualization network
    
    """

    def __init__(
        self,
        device,
        nNeurons,
        nBumps,
        phiFunction=torch.relu,
        wWeight=8.0,
        wShift=2,
        wAttractor=None,
        wScaling=True,
        restingMag=1.0,
        gammaMultiplier=1.0,
        gNoiseMag=None,
        fano=None,
        dt=0.5,
        tau=10.0,
    ):
        """
        Initializing GPU sim object
        
        Inputs
        ------
        device : cuda device name
        nNeurons : number of neurons in each population
        nBumps : number of acitivity bumps
        phiFunction : neural transfer function
        wWeight : inhibition weights
        wShift : left and right populations connectivity shifts
        wScaling : scaling connectivity to be proportional to bump number and network size
        restingMag : magnitude of resting input (A)
        gammaMultiplier : rescaling factor for drive coupling strength
        gNoiseMag : standard deviation of input noise
        fano : Fano factor for spiking noise
        dt : simulation timestep
        tau : time constant
        
        """
        self.device = device  # GPU Name, in pytorch convention
        self.nNeurons = nNeurons  # Number of neurons in each population
        self.nBumps = nBumps  # Number of activity bumps
        self.dt = dt  # timestep
        self.tau = tau  # time constant

        self.phiFunction = phiFunction  # neural activation function

        if wAttractor is not None:  # Check if using predefined connectivity matrix
            self.wAttractor = wAttractor
        else:  # Otherwise creates a new connectivity matrix
            self.wAttractor = utils.generate_w_matrix(
                device,
                nNeurons,
                nBumps,
                wWeight=wWeight,
                wShift=wShift,
                wScaling=wScaling,
            )

        # Quantities used for calculating bump position
        self.positionQuantities = utils.position_quantities(nNeurons, nBumps)

        self.restingInput = restingMag * torch.ones(2 * nNeurons, device=device)
        self.gamma = (  # Coupling strength between input drive and network
            0.1
            * gammaMultiplier
            * torch.cat((-torch.ones(nNeurons), torch.ones(nNeurons))).to(device)
        )

        # Setting input noise magnitude
        self.gNoiseMag = gNoiseMag
        if self.gNoiseMag is not None and self.gNoiseMag <= 0.0:
            self.gNoiseMag = None

        # Setting Fano factor for spiking simulations
        self.fano = fano
        if self.fano is not None and self.fano <= 0.0:
            self.fano = None

    def evolve_network(self, gs, ss, deltaGs, b):
        """
        Evolve network by one timestep
        
        Inputs
        ------
        gs : synaptic inputs, concatenated across populations
        ss : firing rates, concatenated across populations
        deltaGs : updates to gs
        b : driving input
        """
        # Converting synaptic inputs to firing rates
        ss = self.phiFunction(gs)

        if self.fano is not None:  # If running spiking simulations (Eq. 128)
            ss = torch.poisson(ss * self.dt / self.fano) * self.fano / self.dt

        # Main dynamical process (Eq. 125)
        deltaGs = -gs + self.restingInput * (1.0 + self.gamma * b)
        deltaGs += torch.matmul(self.wTotal, ss)
        if self.gNoiseMag is not None:
            deltaGs += self.gNoiseMag * torch.randn_like(gs)

        gs += (self.dt / self.tau) * deltaGs

    def simulate(
        self,
        drive,
        tSim,
        pulseMag=1.0,
        pulsePosition=None,
        wNoise=None,
        saveGs=True,
        trackMotion=False,
        stuckThresh=0.01,
        tSetup=1000,
        tPulse=100,
        tStuck=2000,
        tTrack=200,
        tExtra=1000,
    ):
        """
        Simulate the continuous attractor network
        
        Inputs
        ------
        drive : input drive
        tSim : length of the time of simulation
        pulseMag : magnitude of pulse to help start network at certain positions
        pulsePosition : position at which one bump will be initialized
        wNoise (tensor) : noise in the connectivity matrix. Dim (2 * nNeurons, 2 * nNeurons)
        saveGs : to save the synaptic inputs
        trackMotion : whether to monitor bumps circling the network and getting stuck
        stuckThresh : if network position has not exceeded this value in tStuck time, network is stuck
        tSetup : time for network to initialize into a steady position
        tPulse : time of pulses to initialize network
        tStuck : time to be considered stuck
        tTrack : time range for each round of position detection
        tExtra : extra time at the end of simulation
        """

        device = self.device
        nBumps = self.nBumps
        nNeurons = self.nNeurons
        if isinstance(drive, (int, float)):  # constant drive
            self.drive = np.full(tSetup + tSim, drive)
        else:
            self.drive = drive

        if trackMotion:
            trackNow = True
        else:  # If we are not actively tracking bump motion, we get all positions at the end
            trackNow = False
            tTrack = tSim

        # Allocating memory for tracking data
        trackedGsL = torch.zeros((tTrack, nNeurons), device=device)
        trackedGsR = torch.zeros((tTrack, nNeurons), device=device)

        # Structures for accumulating gs and positions at each timestep
        self.gsL = []
        self.gsR = []
        self.positions = np.empty((0, nBumps))

        self.visited = np.zeros(nNeurons, dtype=int)  # visited positions (rounded to nearest integer)
        self.circled = False  # Check if the network has visited every neural position
        self.stuck = False  # Tracking if network is stuck in same position

        # Additional input for initializing bumps
        bumpPeriod = int(nNeurons / nBumps)  # bump distance
        pulseInds = bumpPeriod * np.arange(nBumps)  # initialize multiple bumps
        pulseInds = np.concatenate((pulseInds, nNeurons + pulseInds))  # both populations
        if pulsePosition is None:
            pulseInds += np.random.randint(bumpPeriod)
        else:
            pulseInds += int(pulsePosition % bumpPeriod)

        pulseInputs = torch.zeros(2 * nNeurons, device=device)
        pulseInputs[pulseInds] = pulseMag

        # Initializing random gs and allocating memory for ss and deltaGs
        gs = 0.1 * torch.rand(2 * nNeurons, device=device) + pulseInputs
        ss = torch.zeros_like(gs)
        deltaGs = torch.zeros_like(gs)

        currentPosition = None

        if wNoise is not None:  # If simulating with connectivity noise
            self.wTotal = self.wAttractor + wNoise
        else:
            self.wTotal = self.wAttractor

        for t in range(tSetup):  # Initialize bumps
            self.evolve_network(gs, ss, deltaGs, self.drive[t])

            if t < tPulse:  # Additional synaptic input at fixed locations
                gs += pulseInputs

        t = 0

        # Main simulation
        while t < tSim:

            # Evolve the network
            self.evolve_network(gs, ss, deltaGs, self.drive[t + tSetup])

            # Save synaptic inputs of left and right populations
            if saveGs:
                self.gsL.append(torch.clone(gs[:nNeurons]))
                self.gsR.append(torch.clone(gs[nNeurons:]))

            trackedGsL[t % tTrack] = torch.clone(gs[:nNeurons])
            trackedGsR[t % tTrack] = torch.clone(gs[nNeurons:])

            # Detecting bump positions and tracking bump motion. This is done intermittently to reduce time transferring tensors to CPU
            if (t + 1) % tTrack == 0:  # Tracking period
                trackedSs = (
                    ((self.phiFunction(trackedGsL) + self.phiFunction(trackedGsR)) / 2)
                    .cpu()
                    .numpy()
                )  # Convert synaptic inputs to activity averaged between populations

                trackedPositions = []
                for ss in trackedSs:  # For all activities tracked, find position and append
                    currentPosition = utils.get_position(
                        ss, nNeurons, nBumps, currentPosition, self.positionQuantities
                    )
                    trackedPositions.append(currentPosition)

                trackedPositions = np.array(trackedPositions)
                self.positions = np.vstack((self.positions, trackedPositions))

                if trackNow:  # If we need to check bumps circling the network or getting stuck
                    trackedNeurons = np.unique(trackedPositions.astype(int))
                    self.visited[trackedNeurons] = 1

                    if (
                        self.visited.sum() == nNeurons
                    ):  # Check if we have visited all neural positions (rounded to integers)
                        self.circled = True
                        trackNow = False
                        t = tSim - tExtra  # Simulate tExtra more time steps
                        continue
                    if t >= tStuck and np.any(
                        np.abs(self.positions[-tStuck] - self.positions[-1])
                        < stuckThresh
                    ):  # Check if any bump has moved less than stuckThresh in the last tStuck timesteps
                        self.stuck = True
                        trackNow = False
                        t = tSim - tExtra  # Simulate tExtra more time steps
                        continue

            t += 1

        if saveGs:
            self.gsL = torch.stack(self.gsL).cpu().numpy()
            self.gsR = torch.stack(self.gsR).cpu().numpy()

    def movie(self, startFrame=-999, endFrame=0, frameInterval=50):

        """
        Method for creating animated movie of neural activites. 
        """
        x = np.arange(self.nNeurons)
        fig = plt.figure(figsize=(8, 6))

        lines = [
            plt.plot([], [], label="{} Neurons".format(["Left", "Right"][i]))[0]
            for i in range(2)
        ]

        yMin = np.min(np.vstack((self.gsL, self.gsR)))
        yMax = np.max(np.vstack((self.gsL, self.gsR)))
        plt.xlim(0, self.nNeurons)
        plt.ylim(yMin, yMax)
        plt.xlabel(
            "Neuron", fontdict={"family": "sans-serif", "color": "black", "size": 27,}
        )
        plt.ylabel(
            "Activity (AU)",
            fontdict={"family": "sans-serif", "color": "black", "size": 27,},
        )
        plt.legend()

        def animate(iFrame):
            leftLine = self.gsL[iFrame]
            rightLine = self.gsR[iFrame]
            lines[0].set_data((x, leftLine))
            lines[1].set_data((x, rightLine))

        anim = FuncAnimation(
            fig, animate, frames=range(startFrame, endFrame), interval=frameInterval
        )
        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()


def find_minimum_speed(
    device,
    nNeurons,
    nBumps,
    wNoiseMag,
    gammaMultiplier=1,
    pulsePosition=None,
    driveDir=1,
    driveMin=0.0,
    driveMax=1.28,
    nSearch=8,
    stuckThresh=0.01,
    tStuck=2000,
    tTrack=200,
    tSim=200000,
    phiFunction=torch.relu,
    wWeight=8.0,
    wScaling=True,
    wNoise=None,
    wNoiseSeed=None,
):

    """
    Function for finding the minimum speed for visiting all neural positions. 
    Performs binary search. 8 rounds of search between 0 and 1.28 will determine
    this minimum speed to a resolution of 0.01
    
    Inputs
    ------
    driveDir : 1 or -1 corresponding to right or left motion
    driveMin : Lower range of search values
    driveMax : Upper range of search values
    nSearch : Number of times to conduct binary search
    
    Returns
    -------
    driveUpper : final drive in the search that enables the bump of activity to travel to every position 
    exploredDrives : sequence of explored drives
    s : GPUSim object containing data of last explored drive
    
    """
    exploredDrives = []
    iSearch = 0  # Current search iteration
    driveMag = driveDir * driveMax  # Start search from upper limit
    driveUpper = driveDir * driveMax  # Upper limit to be searched
    driveLower = driveDir * driveMin  # Lower limit to be searched

    # Generate connectivity matrix
    wAttractor = utils.generate_w_matrix(
        device, nNeurons, nBumps, wWeight=wWeight, wScaling=wScaling
    )

    # Generate noise matrix if not provided
    if wNoise is None:
        if wNoiseSeed is not None:
            torch.manual_seed(wNoiseSeed)
        wNoise = wNoiseMag * torch.randn((2 * nNeurons, 2 * nNeurons), device=device)

    # Create instance of GPU simulation class
    s = GPUSim(
        device,
        nNeurons,
        nBumps,
        phiFunction=phiFunction,
        wAttractor=wAttractor,
        gammaMultiplier=gammaMultiplier,
    )

    # The search process starts with a driveUpper at which the network is circled and a
    # driveLower at which the network is stuck. After the first iteration, the tested
    # driveMag is the average between driveUpper and driveLower. During each iteration,
    # either driveUpper is lowered or driveLower is increased to converge upon the
    # minimum drive required to circle the network.
    while iSearch < nSearch:
        # Simulate network
        s.simulate(
            driveMag,
            tSim,
            wNoise=wNoise,
            pulsePosition=pulsePosition,
            saveGs=False,
            trackMotion=True,
            stuckThresh=stuckThresh,
            tStuck=tStuck,
            tTrack=tTrack,
        )

        exploredDrives.append(driveMag)  # Record drives explored

        # If initial drive is too low for network to be circled, try increasing the upper limit
        if iSearch == 0 and not s.circled:
            driveLower = driveUpper
            driveUpper *= 2
            if driveUpper > driveMax * 2 ** 3:  # Exits if increased more than three times
                print(f"{device} error: Cannot circle at wNoiseMag {wNoiseMag}")
                return np.nan, exploredDrives, s

            driveMag = driveUpper
            print(f"{device} warning: Increasing upper speed to {driveUpper}")
            continue

        if s.circled:  # Network is circled at driveMag
            driveUpper = driveMag  # Decrease driveUpper to driveMag
        else:  # Network is stuck at driveMag
            driveLower = driveMag  # Increase driveLower to driveMag
            if not s.stuck:  # Perhaps increase tSim to better test whether stuck or circled
                print(f"{device} warning: Neither stuck nor circled at speed {driveMag}")

        driveMag = (driveUpper + driveLower) / 2  # Next drive tested is the midpoint
        iSearch += 1

    return driveUpper, exploredDrives, s
