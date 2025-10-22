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


def read_wbse_hyb2_time(fileName):
    """
    Read the time of subroutine hybrid_kernel2 from JSON file.
    """
    with open(fileName, "r") as f:
        raw_ = json.load(f)

    return float(raw_["timing"]["hybrid_k2"]["wall:sec"])


def read_inexact_krylov(fileName):
    """
    Read inexact Krylov threshold from JSON file.
    """

    with open(fileName, "r") as f:
        raw_ = json.load(f)

    return float(raw_["input"]["wbse_control"]["forces_inexact_krylov_tr"])


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

kryl = ["N", "000000001", "00000001", "0000001", "000001", "00001", "0001", "001"]

results = {}

for l in kryl:
    j_file = f"wbse_{l}.json"

    eigs = read_wbse_eigenvalues(j_file)
    wbse_time = read_wbse_time(j_file)
    hyb2_time = read_wbse_hyb2_time(j_file)
    thr = read_inexact_krylov(j_file)
    forces = read_wbse_forces(j_file)

    results[thr] = {}
    results[thr]["eigs"] = list(eigs)
    results[thr]["wbse_time"] = wbse_time
    results[thr]["hyb2_time"] = hyb2_time
    results[thr]["forces"] = {}
    for key in forces:
        results[thr]["forces"][key] = forces[key].tolist()

fname = "results.json"

with open(fname, "w") as f:
    json.dump(results, f, indent=4)
