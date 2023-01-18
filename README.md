# Continuous Attractor Network Simulation and Analysis

This code base is the simulation and analysis code for the paper *Multiple bumps can enhance robustness to noise in continuous attractor networks* by Raymond Wang and Louis Kang. 

[https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010547](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010547)

Our simulation code allows for efficient simulation of a 1-dimensional ring attractor neural network on cuda enabled GPUs and perform various types of analysis.

## Environment Setup
We have included an `environment.yml` file which includes the required packages to run the simulation and analysis code.

These are the dependencies:   
  - python=3.9
  - numpy
  - scipy
  - matplotlib
  - h5py
  - utils
  - IPython
  - torch
  - functools
  - argparse
  - itertools

For optimal performance, our simulation code runs faster on cuda enabled GPUs. 


## Example usage:

*Make sure you are running the code in the same directory as `sim.py`,`analysis.py`, `utils.py`.*

### Simple Simulation Example
For a simple simulation with 100 neurons, 2 bumps of activity: 
```
import sim
s = sim.GPUSim(device="cuda:0", nNeurons=100, nBumps=2)
s.simulate(drive=0.1, tSim=1000)
```

To find the synaptic inputs of the left and right populations: `s.gsL` and `s.gsR` will access them. To find the positions of the bumps of activity `s.positions`.

To run integration analysis to obtain the integration velocities:

```
import analysis
a = analysis.IntegrationAnalysis("cuda:0", s.positions, 100)
```


### Diffusion Analysis Example
For a simulation of a ring attractor with input noise and diffusion analysis:

```
import sim
device = "cuda:0"

tSim = 50000
gNoiseMag = 0.5
drive = 0.5
gammaMultiplier = 1

positions = []
for i in np.arange(20):
    s = sim.GPUSim(device, 100, 1,
                       gNoiseMag=gNoiseMag,
                       gammaMultiplier=gammaMultiplier)
        s.simulate(drive, tSim)
    positions.append(s.positions)
a = analysis.DiffusionAnalysis(device, positions, 100)
```
And to access diffusion constant (`a.diffusionConsts`) and integration velocity (`a.integrationVels`)


### Speed Variation Analysis Example
For a simulation of a ring attractor with connectivity noise and speed variation analysis:

```
import sim
device = "cuda:0"

drive = 1.5
wNoiseMag = 0.002
tSim = 500000
nNeurons = 300
nBumps = 3
gammaMultiplier = 1
device = "cuda:0"

wNoise = wNoiseMag * torch.randn((2*nNeurons,2*nNeurons), device=device)
    
s = sim.GPUSim(device, nNeurons, nBumps,
                   gammaMultiplier=gammaMultiplier)
    
s.simulate(drive, tSim, wNoise=wNoise, trackMotion=True)
aR = analysis.VelocityAnalysis(s)

s.simulate(-drive, tSim, wNoise=wNoise, trackMotion=True)
aL = analysis.VelocityAnalysis(s)

a = analysis.SpeedVariationAnalysis(aR, aL)

```
And to access the speed difference (`a.difference`) and speed variability (`a.variability`)

