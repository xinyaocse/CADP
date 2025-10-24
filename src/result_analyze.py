import math
import numpy as np


def result_analyze(attack_result, noisy_data, org_data, mechanism, scale, threshold):
    dim_count = org_data.shape[1]

    # Attack result
    amse_ptb = np.mean(np.sum((noisy_data - org_data) ** 2, axis=1) / dim_count)
    amse_atk = np.mean(np.sum((attack_result - org_data) ** 2, axis=1) / dim_count)
    aes = (1 - (amse_atk / amse_ptb)) * 100

    # Privacy result
    # (Assuming that data are normalized)

    # Traditional DP
    dp = calculate_dp_epsilon(org_data, mechanism, scale)
    # CDP
    cdp = calculate_cdp_epsilon(org_data, dp, threshold)
    # DDP
    ddp = calculate_ddp_epsilon(org_data, dp, threshold)
    # BDP
    bdp = calculate_bdp_epsilon(org_data, dp, threshold)
    # CADP ('Gauss' and 'CorrGauss' only)
    if mechanism == 'gauss' or mechanism == 'cgm':
        cadp = calculate_cadp_epsilon(attack_result, noisy_data, org_data, dp)
    else:
        cadp = None

    return amse_ptb, amse_atk, aes, dp, cdp, ddp, bdp, cadp


def calculate_dp_epsilon(org_data, mechanism, scale, delta=1e-5):
    dim_count = org_data.shape[1]

    if isinstance(scale, (int, float)) and not isinstance(scale, bool):
        scale = np.ones(dim_count) * scale
    elif isinstance(scale, list):
        scale = np.array(scale)
    elif isinstance(scale, np.ndarray):
        scale = scale

    if mechanism == 'gauss' or mechanism == 'cgm':
        return np.sqrt(2 * np.log(1.25 / delta)) / scale
    elif mechanism == 'lap':
        return 1 / scale


def calculate_cadp_epsilon(attack_result, noisy_data, org_data, dp_epsilon):
    noisy_dis = np.mean(np.abs((noisy_data - org_data)), axis=0)
    attack_dis = np.mean(np.abs((attack_result - org_data)), axis=0)
    ratio = noisy_dis / attack_dis
    return dp_epsilon * ratio


def calculate_cdp_epsilon(org_data, dp_epsilon, threshold):
    dim_count = org_data.shape[1]
    abs_corr = np.abs(np.corrcoef(org_data, rowvar=False))
    cdp_epsilon = np.zeros(dim_count)
    for i in range(dim_count):
        for j in range(dim_count):
            if abs_corr[i, j] >= threshold:
                cdp_epsilon[i] += abs_corr[i, j] * dp_epsilon[j]
    return cdp_epsilon


def calculate_ddp_epsilon(org_data, dp_epsilon, threshold):
    dim_count = org_data.shape[1]
    abs_corr = np.abs(np.corrcoef(org_data, rowvar=False))
    rho = np.eye(dim_count)

    for i in range(dim_count):
        for j in range(dim_count):
            if i == j:
                continue
            if abs_corr[i, j] >= threshold:
                rho[i, j] = get_rho_ddp(org_data[:, i], org_data[:, j], dp_epsilon[i])

    ddp_epsilon = np.zeros(dim_count)
    for i in range(dim_count):
        for j in range(dim_count):
            if abs_corr[i, j] >= threshold:
                ddp_epsilon[i] += rho[i, j] * dp_epsilon[j]
    return ddp_epsilon


def get_rho_ddp(d1, d2, epsilon1, b=100):
    # calculate ρ that d2 contributes to d1

    # We use discrete variables to fit continuous variables.
    hist, x_edges, y_edges = np.histogram2d(d1, d2, bins=b)
    max = np.max(d1)
    min = np.min(d1)
    mid = (max + min) / 2
    dep_indist = np.zeros(b)
    # i : d2
    for i in range(b):
        # j : d1
        for j in range(b):
            if hist[j][i] != 0:
                dep_indist[i] += hist[j][i] / np.sum(hist[:, i]) * math.exp(
                    get_dist_ddp((x_edges[j] + x_edges[j + 1]) / 2, max, min, mid) / (max - min) * epsilon1)

    return math.log(np.max(dep_indist)) / epsilon1


def get_dist_ddp(x, max, min, mid):
    if x <= mid:
        return max - x
    else:
        return x - min


def calculate_bdp_epsilon(org_data, dp_epsilon, threshold):
    # When the adversary has no prior knowledge on the real value of raw data
    dim_count = org_data.shape[1]
    abs_corr = np.abs(np.corrcoef(org_data, rowvar=False))
    bdp_epsilon = np.zeros(dim_count)
    for i in range(dim_count):
        for j in range(dim_count):
            if abs_corr[i, j] >= threshold:
                bdp_epsilon[i] += dp_epsilon[j]
    return bdp_epsilon
