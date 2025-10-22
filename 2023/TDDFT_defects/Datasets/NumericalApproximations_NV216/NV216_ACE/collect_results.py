import numpy as np
import json
import subprocess


def read_wbse_eigenvalues(fileName):
    """
    Read eigenvalues from JSON file.
    """

    with open(fileName, "r") as f:
        raw_ = json.load(f)

    return np.array(raw_["exec"]["davitr"][-1]["ev"], dtype=float)


def read_wbse_time(fileName):
    """
    Read the total wall time from JSON file.
    """
    with open(fileName, "r") as f:
        raw_ = json.load(f)

    return float(raw_["timing"]["WBSE"]["wall:sec"])


def read_wbse_hpsi_time(fileName):
    """
    Read the time of subroutine h_psi from JSON file.
    """
    with open(fileName, "r") as f:
        raw_ = json.load(f)

    return float(raw_["timing"]["h_psi"]["wall:sec"])


def read_wbse_forces(fileName):
    """
    Read forces from JSON file.
    """

    with open(fileName, "r") as f:
        raw_ = json.load(f)

    forces = {}
    for key in raw_["exec"]["forces"]:
        forces[key] = np.array(raw_["exec"]["forces"][key], dtype=float)

    return forces


########
# Main #
########

nace = ["N", "2586", "2155", "1724", "1293", "862", "762", "662", "562", "462"]

results = {}

for l in nace:
    j_file = f"wbse_{l}.json"

    eigs = read_wbse_eigenvalues(j_file)
    wbse_time = read_wbse_time(j_file)
    hpsi_time = read_wbse_hpsi_time(j_file)
    forces = read_wbse_forces(j_file)

    results[l] = {}
    results[l]["eigs"] = list(eigs)
    results[l]["wbse_time"] = wbse_time
    results[l]["hpsi_time"] = hpsi_time
    results[l]["forces"] = {}
    for key in forces:
        results[l]["forces"][key] = forces[key].tolist()

fname = "results.json"

with open(fname, "w") as f:
    json.dump(results, f, indent=4)
