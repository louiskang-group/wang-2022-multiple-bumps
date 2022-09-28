import numpy as np
import torch
import time


# numpy ReLU
def relu(x):
    return np.multiply(x, x > 0)


def generate_w_matrix(device, nNeurons, nBumps, wWeight=8.0, wShift=2, wScaling=True):
    """
    Generating synaptic connectivity matrix
    
    Inputs
    ------
    wWeight : positive value; sets the strength of the most inhibitory connection
    wShift : synaptic output shift for L and R populations (xi)
    wScaling : scale the raw wWeight by nNeurons and nBumps
    """
    # Calculating synaptic connectivity values
    length = nNeurons / (
        2.28 * nBumps
    )  # inhibition length l that produces nBumps (Eq. 47)
    length2 = int(2 * np.ceil(length))
    positions = np.arange(-length2, length2 + 1)  # Only short-range synapses between -2l and 2l
    if wScaling:  # Scale wWeight so bump shape remains the same
        strength = wWeight * nBumps / nNeurons
    else:
        strength = wWeight

    # Cosine-based connectivity function (Eq. 38)
    values = strength * (np.cos((np.pi * positions / length)) - 1) / 2
    values *= np.abs(positions) < 2 * length

    # Adding values to form unshifted row of w matrix. We form the row this way so that
    # synaptic weights are wrapped around the network in case 4 * length > nNeurons
    # (Eq. 127)
    wUnshifted = torch.zeros(nNeurons, device=device)
    for position, w in zip(positions, values):
        wUnshifted[position % nNeurons] += w

    # Form unshifted matrix of dim (nNeurons, nNeurons), then shift and form final matrix
    # of dim (2 * nNeurons, 2 * nNeurons)
    wQuadrant = torch.vstack([wUnshifted.roll(i) for i in range(nNeurons)])
    wMatrix = torch.hstack((wQuadrant.roll(-wShift, 0), wQuadrant.roll(wShift, 0)))
    wMatrix = torch.vstack((wMatrix, wMatrix))

    return wMatrix


def position_quantities(nNeurons, nBumps):
    
    """
    Calculate constant quantities used repeatedly to calculate position
    """

    # Position calculation requires extraction of each bump in regions of length
    # splitLength. Neurons at the midway points between bumps may not be included
    # into a region. splitMask indicates which neural positions should not be used,
    # distributing them as evenly as possible throughout the network. For example,
    # for a network with 200 neurons (per population) and 3 bumps has regions of
    # 66 neurons with 2 neural positions unused.
    splitLength = int(np.floor(nNeurons / nBumps))
    splitRemainder = nNeurons % nBumps

    splitExtras = (nNeurons - 1) - np.arange(splitRemainder) * splitLength

    splitMask = np.ones(nNeurons, dtype=bool)
    splitMask[splitExtras] = False

    # centeredOffsets contains the positions of the first neuron in each region
    centeredOffsets = np.arange(nBumps) * splitLength
    if splitRemainder > 1:
        for i in range(splitRemainder - 1):
            centeredOffsets[-(i + 1) :] += 1

    # Converting neural positions to angles that obey the periodicity of the bumps
    # for calculating the circular mean
    angles = 2 * np.pi * nBumps / nNeurons * np.arange(nNeurons)
    cosAngles = np.cos(angles)
    sinAngles = np.sin(angles)

    return (splitMask, centeredOffsets, cosAngles, sinAngles)


def get_position(ss, nNeurons, nBumps, old, quantities):
    
    """
    Find the position of bumps

    Inputs
    ------
    ss : firing rates averaged between populations
    old : old bump positions for matching
    quantities : constants produced by position_quantities

    """

    splitLength = int(np.floor(nNeurons / nBumps))
    splitMask = quantities[0]

    # Compute the circular center of mass to calculate shift, which approximately
    # centers bumps within each extraction region. circularCenter is an average position
    # of all the bumps.
    circularCenter = (
        nNeurons
        / (2 * np.pi * nBumps)
        * circular_mean(ss, quantities[2], quantities[3])
    )
    shift = int(splitLength / 2 - circularCenter)

    
    # Shift activity so position circularCenter becomes position splitLength/2,
    # then apply mask and reshape to extract nBumps regions of length splitLength
    centeredSs = np.roll(ss, shift)[splitMask].reshape((nBumps, -1))
    centeredShifts = quantities[1] - shift

    # Compute linear center of mass for each region and undo shift to determine bump
    # positions
    centers = np.sum(
        centeredSs * np.expand_dims(np.arange(splitLength), 0), 1
    ) / np.sum(centeredSs, 1)
    centers = np.mod(centers + centeredShifts, nNeurons)

    # Reorder bumps if they have moved from one region to the next so that their
    # identities are maintained. This is done by comparing old and positions to
    # determine the ordering that minimizes their mean distance
    if nBumps > 1 and old is not None:
        match = int(
            np.round(
                np.mean(mod_offset((centers - old) / (nNeurons / nBumps), nBumps, -0.5))
            )
        )
        centers = np.roll(centers, match)

    return centers


# mod function with lower range starting at offset
def mod_offset(x, n, offset):
    return (x - offset) % n + offset


# \arctan [(\sum_i w_i \sin\theta_i) / (\sum_i w_i \cos\theta_i)]
# Then shift result to lie between 0 and 2 \pi
def circular_mean(weights, cosAngles, sinAngles):

    return np.mod(
        np.arctan2((weights * sinAngles).sum(), (weights * cosAngles).sum()), 2 * np.pi
    ).item()


# Timing utilities
start_time = None

def start_timer():
    global start_time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    print("\nTotal execution time {:.3f} sec".format(end_time - start_time))
