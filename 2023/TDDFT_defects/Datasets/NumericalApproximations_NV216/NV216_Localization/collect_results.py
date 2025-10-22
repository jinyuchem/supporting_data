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


def read_wbse_kernel_time(fileName):
    """
    Read the time of subroutine h_psi from JSON file.
    """
    with open(fileName, "r") as f:
        raw_ = json.load(f)

    return float(raw_["timing"]["bse_kernel"]["wall:sec"])


def read_localization(fileName):
    """
    Read localization from JSON file.
    """

    with open(fileName, "r") as f:
        raw_ = json.load(f)

    return float(raw_["input"]["wbse_init_control"]["overlap_thr"])


def read_npairs(fileName, nspin=2):
    """
    Read number of pairs from JSON file.
    """

    text = subprocess.getstatusoutput(f"grep n_pairs {fileName} -A 2 | tail -1")[
        1
    ].split()[-1]
    return int(text)

    # When using spin parallelization, n_pairs for the spin down channel is not printed out.
    # But n_pairs from both spin channels should always be close to one another.


#    if nspin == 2:
#        text = subprocess.getstatusoutput("grep '%s' %s -A %d | tail -1"%(
#               'n_pairs', fileName, 2))[1][-8:]
#        n_pairs.append(int(text))


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

local = ["N", "00001", "0001", "001", "01"]

results = {}

for l in local:
    j_file = f"wbse_{l}.json"
    o_file = f"wbse_init_{l}.out"

    eigs = read_wbse_eigenvalues(j_file)
    wbse_time = read_wbse_time(j_file)
    kernel_time = read_wbse_kernel_time(j_file)
    ovlpthr = read_localization(j_file)
    n_pairs = read_npairs(o_file)
    forces = read_wbse_forces(j_file)

    results[ovlpthr] = {}
    results[ovlpthr]["eigs"] = list(eigs)
    results[ovlpthr]["wbse_time"] = wbse_time
    results[ovlpthr]["kernel_time"] = kernel_time
    results[ovlpthr]["n_pairs"] = n_pairs
    results[ovlpthr]["forces"] = {}
    for key in forces:
        results[ovlpthr]["forces"][key] = forces[key].tolist()

fname = "results.json"

with open(fname, "w") as f:
    json.dump(results, f, indent=4)
