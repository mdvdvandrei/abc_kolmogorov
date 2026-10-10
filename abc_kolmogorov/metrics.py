"""Field scores for a pinned ABC forecast.

RMSE of the ensemble mean and of a member, fair CRPS, spread, enstrophy,
and the fraction of power at k > 8. Persistence holds the initial frame.
"""

from __future__ import annotations

import numpy as np


def crps_field(samples: np.ndarray, truth: np.ndarray) -> float:
    """Fair empirical CRPS averaged over the field. samples (M, H, W), truth (H, W)."""
    members = samples.shape[0]
    term1 = np.mean(np.abs(samples - truth), axis=0)
    order = np.sort(samples, axis=0)
    coeff = (2 * np.arange(1, members + 1) - members - 1).astype(np.float64)
    coeff = coeff.reshape((members,) + (1,) * (order.ndim - 1))
    term2 = np.sum(coeff * order, axis=0) / (members**2)
    return float(np.mean(term1 - term2))


def field_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def enstrophy(q: np.ndarray) -> float:
    return float(0.5 * np.mean(q**2))


def high_k_fraction(field: np.ndarray, kmin: float = 8.0) -> float:
    power = np.abs(np.fft.fft2(field)) ** 2
    n = field.shape[-1]
    freq = np.fft.fftfreq(n, d=1.0 / n)
    rad = np.sqrt(freq[:, None] ** 2 + freq[None, :] ** 2)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    return float(power[rad > kmin].sum() / total)


def score_leads(
    samples_n: np.ndarray,
    truth_n: np.ndarray,
    leads: list[int],
    mean: float,
    std: float,
) -> dict:
    """samples_n (N, M, T, H, W) and truth_n (N, T, H, W), both normalized."""
    samples = samples_n * std + mean
    truth = truth_n * std + mean
    rmse_mean = []
    rmse_member = []
    crps = []
    spread = []
    persist = []
    z_true = []
    z_mean = []
    z_member = []
    hk_true = []
    hk_member = []
    for lead in leads:
        err_mean = []
        err_member = []
        crps_l = []
        spread_l = []
        persist_l = []
        zt = []
        zm = []
        zmem = []
        hkt = []
        hkm = []
        for n in range(truth.shape[0]):
            members = samples[n, :, lead]
            target = truth[n, lead]
            ens = members.mean(axis=0)
            err_mean.append(field_rmse(ens, target))
            err_member.append(float(np.mean([field_rmse(m, target) for m in members])))
            crps_l.append(crps_field(members, target))
            spread_l.append(float(np.sqrt(np.mean(np.var(members, axis=0)))))
            persist_l.append(field_rmse(truth[n, 0], target))
            zt.append(enstrophy(target))
            zm.append(enstrophy(ens))
            zmem.append(float(np.mean([enstrophy(m) for m in members])))
            hkt.append(high_k_fraction(target))
            hkm.append(float(np.mean([high_k_fraction(m) for m in members])))
        rmse_mean.append(float(np.mean(err_mean)))
        rmse_member.append(float(np.mean(err_member)))
        crps.append(float(np.mean(crps_l)))
        spread.append(float(np.mean(spread_l)))
        persist.append(float(np.mean(persist_l)))
        z_true.append(float(np.mean(zt)))
        z_mean.append(float(np.mean(zm)))
        z_member.append(float(np.mean(zmem)))
        hk_true.append(float(np.mean(hkt)))
        hk_member.append(float(np.mean(hkm)))
    return {
        "lead_frames": leads,
        "rmse_ensemble_mean": rmse_mean,
        "rmse_member": rmse_member,
        "crps": crps,
        "spread": spread,
        "rmse_persistence": persist,
        "si_ensemble_mean": [r / max(z * 2, 1e-8) ** 0.5 for r, z in zip(rmse_mean, z_true)],
        "si_persistence": [r / max(z * 2, 1e-8) ** 0.5 for r, z in zip(persist, z_true)],
        "enstrophy_truth": z_true,
        "enstrophy_ensemble_mean": z_mean,
        "enstrophy_member": z_member,
        "high_k_truth": hk_true,
        "high_k_member": hk_member,
    }
