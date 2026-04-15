from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

Array = np.ndarray


def _rng(random_state: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(random_state)


def _softmax(z: Array) -> Array:
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def _calibrate_softmax_intercepts(
    eta_no_b: Array,
    class_probs: Array,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Array:
    """Find intercepts b so mean softmax(eta_no_b + b) ~= class_probs."""
    target = np.asarray(class_probs, dtype=float)
    target = target / np.maximum(target.sum(), 1e-12)
    b = np.zeros(target.shape[0], dtype=float)

    for _ in range(max_iter):
        p = _softmax(eta_no_b + b)
        emp = np.clip(p.mean(axis=0), 1e-12, 1.0)
        delta = np.log(np.clip(target, 1e-12, 1.0)) - np.log(emp)
        b = b + delta
        b = b - b.mean()
        if np.max(np.abs(delta)) < tol:
            break
    return b


def _build_support(
    n_features: int,
    n_informative: int,
    rng: np.random.Generator,
    support_mask: Optional[Array] = None,
) -> Array:
    if support_mask is not None:
        support = np.asarray(support_mask, dtype=bool)
        if support.shape[0] != n_features:
            raise ValueError("support_mask debe tener longitud n_features")
        return support

    k = int(np.clip(n_informative, 0, n_features))
    support = np.zeros(n_features, dtype=bool)
    if k > 0:
        idx = rng.choice(n_features, size=k, replace=False)
        support[idx] = True
    return support


def _scale_noise_for_snr(signal: Array, snr: float = 5.0) -> float:
    var_signal = float(np.var(signal))
    if snr <= 0 or not np.isfinite(snr):
        return float(np.sqrt(max(var_signal, 1e-12)))
    sigma2 = var_signal / max(float(snr), 1e-12)
    return float(np.sqrt(max(sigma2, 1e-12)))


def _sample_gaussian_features(
    n_samples: int,
    n_features: int,
    corr: str,
    rho: float,
    n_blocks: int,
    rng: np.random.Generator,
) -> Array:
    corr = str(corr).lower()

    if corr == "independent":
        return rng.standard_normal((n_samples, n_features))

    rho = float(np.clip(rho, -0.99, 0.99))
    idx = np.arange(n_features)

    if corr == "toeplitz":
        cov = rho ** np.abs(np.subtract.outer(idx, idx))
    elif corr == "block":
        cov = np.eye(n_features)
        n_blocks = max(1, int(n_blocks))
        block_sizes = [n_features // n_blocks] * n_blocks
        for i in range(n_features % n_blocks):
            block_sizes[i] += 1

        start = 0
        for bs in block_sizes:
            end = start + bs
            if bs > 1:
                cov[start:end, start:end] = rho
                np.fill_diagonal(cov[start:end, start:end], 1.0)
            start = end
    else:
        raise ValueError("corr debe ser 'independent', 'toeplitz' o 'block'")

    # Jitter for numerical stability in Cholesky.
    cov = cov + np.eye(n_features) * 1e-8
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_samples, n_features))
    return z @ L.T


def _add_redundant_features(
    X: Array,
    support_idx: Array,
    n_redundant: int,
    noise_std: float,
    rng: np.random.Generator,
) -> Tuple[Array, List[Tuple[int, List[int]]]]:
    n, p = X.shape
    if n_redundant <= 0 or support_idx.size == 0:
        return X, []

    new_cols = []
    mapping: List[Tuple[int, List[int]]] = []

    for j in range(int(n_redundant)):
        m = int(min(max(1, support_idx.size), rng.integers(1, min(4, support_idx.size) + 1)))
        src = rng.choice(support_idx, size=m, replace=False)
        w = rng.normal(0.0, 1.0, size=m)
        col = X[:, src] @ w + rng.normal(0.0, noise_std, size=n)
        new_cols.append(col)
        mapping.append((p + j, src.tolist()))

    X2 = np.column_stack([X, np.column_stack(new_cols)])
    return X2, mapping


def _add_repeated_features(
    X: Array,
    n_repeated: int,
    rng: np.random.Generator,
) -> Tuple[Array, List[int]]:
    n, p = X.shape
    if n_repeated <= 0 or p == 0:
        return X, []

    src = rng.integers(0, p, size=int(n_repeated))
    repeated = X[:, src]
    X2 = np.column_stack([X, repeated])
    return X2, src.tolist()


def _add_categorical_noise(
    X: Array,
    n_categorical: int,
    n_categories: int,
    rng: np.random.Generator,
) -> Tuple[Array, List[int]]:
    if n_categorical <= 0:
        return X, []
    n = X.shape[0]
    n_categories = max(2, int(n_categories))
    cat = rng.integers(0, n_categories, size=(n, int(n_categorical)))
    start = X.shape[1]
    X2 = np.column_stack([X, cat.astype(float)])
    return X2, list(range(start, X2.shape[1]))


def _sample_nb_means(
    p: int,
    rng: np.random.Generator,
    mean_log_expr: float,
    sd_log_expr: float,
) -> Array:
    vals = np.exp(rng.normal(mean_log_expr, sd_log_expr, size=p))
    return vals / np.maximum(vals.sum(), 1e-12)


def _sample_library_sizes(
    n_samples: int,
    rng: np.random.Generator,
    log_mean: float,
    log_sd: float,
) -> Array:
    return np.exp(rng.normal(log_mean, log_sd, size=n_samples))


def _sample_nb_counts(mu: Array, alpha_g: Array, rng: np.random.Generator) -> Array:
    alpha = np.clip(np.asarray(alpha_g, dtype=float), 1e-10, None)[None, :]
    shape = 1.0 / alpha
    scale = alpha * np.maximum(mu, 1e-12)
    lam = rng.gamma(shape=shape, scale=scale)
    return rng.poisson(lam)


def _pick_affected_features(
    p: int,
    fraction: float,
    rng: np.random.Generator,
) -> Array:
    if p <= 0:
        return np.array([], dtype=int)
    frac = float(np.clip(fraction, 0.0, 1.0))
    if frac <= 0:
        return np.array([], dtype=int)
    if frac >= 1.0:
        return np.arange(p)
    k = max(1, int(round(frac * p)))
    return np.sort(rng.choice(p, size=k, replace=False))


def _assign_batches(
    n_samples: int,
    n_batches: int,
    rng: np.random.Generator,
    batch_probs: Optional[Sequence[float]] = None,
    y: Optional[Array] = None,
    task: str = "classification",
    batch_confounded_with_label: float = 0.0,
    n_classes: int = 2,
) -> Tuple[Optional[Array], Array, Dict[str, Any]]:
    if n_batches <= 0:
        return None, np.array([], dtype=float), {"enabled": False}

    b = int(n_batches)
    if batch_probs is None:
        probs = np.full(b, 1.0 / b, dtype=float)
    else:
        probs = np.asarray(batch_probs, dtype=float)
        if probs.shape[0] != b:
            raise ValueError("batch_probs debe tener longitud n_batches")
        probs = probs / np.maximum(probs.sum(), 1e-12)

    conf = float(np.clip(batch_confounded_with_label, 0.0, 1.0))
    y_in = y is not None and str(task) == "classification"
    conf_enabled = y_in and conf > 0.0

    if conf_enabled:
        y_int = np.asarray(y, dtype=int)
        class_batch = np.tile(probs[None, :], (max(1, int(n_classes)), 1))
        for k in range(class_batch.shape[0]):
            focus = np.zeros(b, dtype=float)
            focus[k % b] = 1.0
            class_batch[k] = (1.0 - conf) * probs + conf * focus
            class_batch[k] = class_batch[k] / np.maximum(class_batch[k].sum(), 1e-12)

        batch_id = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            cls = int(y_int[i])
            cls = min(max(cls, 0), class_batch.shape[0] - 1)
            batch_id[i] = int(rng.choice(b, p=class_batch[cls]))
    else:
        batch_id = rng.choice(b, size=n_samples, p=probs)

    meta = {
        "enabled": True,
        "n_batches": b,
        "batch_probs": probs,
        "batch_confounded_with_label": conf,
        "confounding_active": bool(conf_enabled),
    }
    return batch_id.astype(int), probs, meta


def _apply_batch_effects(
    X: Array,
    batch_id: Optional[Array],
    n_batches: int,
    batch_feature_shift: float,
    batch_feature_scale: float,
    batch_effect_on_subset: float,
    rng: np.random.Generator,
) -> Tuple[Array, Dict[str, Any]]:
    if batch_id is None or n_batches <= 0:
        return X, {"enabled": False}

    shift_mag = float(batch_feature_shift)
    scale_mag = float(batch_feature_scale)
    if shift_mag == 0.0 and scale_mag == 0.0:
        return X, {
            "enabled": True,
            "n_batches": int(n_batches),
            "shift": 0.0,
            "scale": 0.0,
            "affected_feature_fraction": float(batch_effect_on_subset),
            "affected_features": np.array([], dtype=int),
        }

    n, p = X.shape
    frac = 1.0 if batch_effect_on_subset is None else float(np.clip(batch_effect_on_subset, 0.0, 1.0))
    affected = _pick_affected_features(p, frac, rng)
    if affected.size == 0 and p > 0:
        affected = np.arange(p)

    shift_vectors = np.zeros((n_batches, p), dtype=float)
    scale_vectors = np.ones((n_batches, p), dtype=float)

    if shift_mag != 0.0:
        shift_vectors[:, affected] = rng.normal(0.0, shift_mag, size=(n_batches, affected.size))
    if scale_mag != 0.0:
        raw_scale = rng.normal(0.0, scale_mag, size=(n_batches, affected.size))
        scale_vectors[:, affected] = np.clip(1.0 + raw_scale, 0.05, 20.0)

    X2 = X.copy()
    for b in range(int(n_batches)):
        m = batch_id == b
        if not np.any(m):
            continue
        X2[m] = X2[m] * scale_vectors[b] + shift_vectors[b]

    meta = {
        "enabled": True,
        "n_batches": int(n_batches),
        "shift": shift_mag,
        "scale": scale_mag,
        "affected_feature_fraction": frac,
        "affected_feature_count": int(affected.size),
        "affected_features_preview": affected[:50],
        "shift_std": float(np.std(shift_vectors[:, affected])) if affected.size > 0 else 0.0,
        "scale_std": float(np.std(scale_vectors[:, affected])) if affected.size > 0 else 0.0,
    }
    return X2, meta


def _apply_latent_confounders(
    X: Array,
    y: Array,
    info: Dict[str, Any],
    latent_confounders: int,
    latent_strength: float,
    latent_link_to_label: float,
    rng: np.random.Generator,
) -> Tuple[Array, Array, Dict[str, Any]]:
    k = int(max(0, latent_confounders))
    sx = float(latent_strength)
    sy = float(latent_link_to_label)

    if k <= 0 or (sx == 0.0 and sy == 0.0):
        return X, y, {"enabled": False}

    n, p = X.shape
    U = rng.normal(0.0, 1.0, size=(n, k))
    X2 = X.copy()
    y2 = y.copy()

    wx = None
    if sx != 0.0:
        wx = rng.normal(0.0, sx / np.sqrt(max(k, 1)), size=(k, p))
        X2 = X2 + U @ wx

    if sy != 0.0:
        if info.get("task") == "classification":
            n_classes = int(info.get("n_classes", np.max(y2) + 1))
            wy = rng.normal(0.0, sy / np.sqrt(max(k, 1)), size=(k, n_classes))
            logits = U @ wy
            p_u = _softmax(logits)
            cum = np.cumsum(p_u, axis=1)
            r = rng.random(n)[:, None]
            y_u = (cum < r).sum(axis=1)
            mix_prob = float(np.clip(abs(sy), 0.0, 1.0))
            m = rng.random(n) < mix_prob
            y2[m] = y_u[m]
        else:
            wy = rng.normal(0.0, sy / np.sqrt(max(k, 1)), size=k)
            y2 = y2 + U @ wy

    if U.size <= 5000:
        latent_u = U
    else:
        latent_u = {
            "shape": [int(U.shape[0]), int(U.shape[1])],
            "mean": float(U.mean()),
            "std": float(U.std()),
        }

    meta = {
        "enabled": True,
        "latent_confounders": k,
        "latent_strength": sx,
        "latent_link_to_label": sy,
        "latent_U": latent_u,
        "latent_U_shape": [int(U.shape[0]), int(U.shape[1])],
    }
    return X2, y2, meta


def _parse_shift_splits(shift_applies_to: str) -> List[str]:
    key = str(shift_applies_to).lower().strip()
    if key == "val":
        return ["val"]
    if key == "test":
        return ["test"]
    if key == "valtest":
        return ["val", "test"]
    raise ValueError("shift_applies_to debe ser 'val', 'test' o 'valtest'")


def _validate_split(split: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if len(split) != 3:
        raise ValueError("split debe tener 3 valores (train, val, test)")
    a, b, c = [float(x) for x in split]
    if min(a, b, c) <= 0:
        raise ValueError("split debe contener fracciones positivas")
    s = a + b + c
    if not np.isfinite(s) or s <= 0:
        raise ValueError("split inválido")
    return a / s, b / s, c / s


def _split_indices(
    n_samples: int,
    split: Tuple[float, float, float],
    rng: np.random.Generator,
) -> Dict[str, Array]:
    tr, va, te = _validate_split(split)
    perm = rng.permutation(n_samples)

    n_tr = int(np.floor(tr * n_samples))
    n_va = int(np.floor(va * n_samples))
    n_te = n_samples - n_tr - n_va

    if n_tr <= 0 or n_va <= 0 or n_te <= 0:
        raise ValueError("split produce particiones vacías; ajusta n_samples o split")

    idx_tr = perm[:n_tr]
    idx_va = perm[n_tr:n_tr + n_va]
    idx_te = perm[n_tr + n_va:]
    return {"train": idx_tr, "val": idx_va, "test": idx_te}


def _estimate_class_scores(X: Array, y: Array, n_classes: int) -> Array:
    """Simple class-score proxy using class centroids."""
    n_classes = int(max(2, n_classes))
    p = X.shape[1]
    means = np.zeros((n_classes, p), dtype=float)
    for c in range(n_classes):
        m = y == c
        if np.any(m):
            means[c] = X[m].mean(axis=0)
    return X @ means.T


def _resample_to_prior(
    X: Array,
    y: Array,
    target_probs: Array,
    rng: np.random.Generator,
    extra_arrays: Optional[Dict[str, Array]] = None,
) -> Tuple[Array, Array, Dict[str, Array]]:
    n = y.shape[0]
    n_classes = target_probs.shape[0]
    counts = rng.multinomial(n, target_probs)

    idx_new = []
    for c in range(n_classes):
        pool = np.flatnonzero(y == c)
        if pool.size == 0:
            pool = np.arange(n)
        take = counts[c]
        chosen = rng.choice(pool, size=take, replace=(pool.size < take))
        idx_new.append(chosen)

    idx = np.concatenate(idx_new)
    rng.shuffle(idx)

    out_extra: Dict[str, Array] = {}
    if extra_arrays:
        for k, arr in extra_arrays.items():
            out_extra[k] = arr[idx]

    return X[idx], y[idx], out_extra


def _apply_domain_shift(
    split_data: Dict[str, Dict[str, Array]],
    info: Dict[str, Any],
    domain_shift: str,
    domain_shift_strength: float,
    shift_applies_to: str,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    mode = str(domain_shift).lower()
    strength = float(domain_shift_strength)

    if mode == "none" or strength == 0.0:
        return {
            "mode": "none",
            "strength": strength,
            "shifted_splits": [],
            "details": {},
        }

    targets = _parse_shift_splits(shift_applies_to)
    details: Dict[str, Any] = {}

    if mode == "covariate":
        for name in targets:
            Xs = split_data[name]["X"]
            p = Xs.shape[1]
            delta = rng.normal(0.0, strength, size=p)
            scale = np.clip(1.0 + rng.normal(0.0, 0.5 * strength, size=p), 0.1, 10.0)
            split_data[name]["X"] = Xs * scale + delta
            details[name] = {
                "mean_shift_norm": float(np.linalg.norm(delta)),
                "scale_mean": float(np.mean(scale)),
                "scale_std": float(np.std(scale)),
            }

    elif mode == "prior":
        if info.get("task") != "classification":
            details["warning"] = "prior shift ignorado: task no es clasificación"
        else:
            k = int(info.get("n_classes", 2))
            base = np.asarray(info.get("class_probs", np.full(k, 1.0 / k)), dtype=float)
            base = base / np.maximum(base.sum(), 1e-12)
            for name in targets:
                perturb = rng.normal(0.0, strength, size=k)
                target = np.exp(np.log(np.clip(base, 1e-9, 1.0)) + perturb)
                target = target / np.maximum(target.sum(), 1e-12)

                extra = {
                    "batch_id": split_data[name].get("batch_id"),
                    "clean_y": split_data[name].get("clean_y"),
                }
                extra = {kk: vv for kk, vv in extra.items() if vv is not None}

                Xn, yn, extra_out = _resample_to_prior(
                    split_data[name]["X"],
                    split_data[name]["y"],
                    target,
                    rng,
                    extra_arrays=extra,
                )
                split_data[name]["X"] = Xn
                split_data[name]["y"] = yn
                if "batch_id" in extra_out:
                    split_data[name]["batch_id"] = extra_out["batch_id"]
                if "clean_y" in extra_out:
                    split_data[name]["clean_y"] = extra_out["clean_y"]
                details[name] = {
                    "target_class_probs": target,
                    "actual_class_probs": np.bincount(yn.astype(int), minlength=k) / max(1, yn.size),
                }

    elif mode == "concept":
        for name in targets:
            Xs = split_data[name]["X"]
            ys = split_data[name]["y"]

            if info.get("task") == "classification":
                k = int(info.get("n_classes", 2))
                base_w = info.get("coef_matrix", None)
                if isinstance(base_w, np.ndarray) and base_w.shape[0] == Xs.shape[1] and base_w.shape[1] == k:
                    w = base_w + rng.normal(0.0, strength, size=base_w.shape)
                else:
                    w = rng.normal(0.0, 1.0, size=(Xs.shape[1], k))
                logits = Xs @ w
                pri = np.bincount(ys.astype(int), minlength=k)
                pri = pri / np.maximum(pri.sum(), 1)
                b = _calibrate_softmax_intercepts(logits, pri)
                p = _softmax(logits + b)
                cum = np.cumsum(p, axis=1)
                r = rng.random(Xs.shape[0])[:, None]
                y_new = (cum < r).sum(axis=1)
                split_data[name]["y"] = y_new.astype(int)
                details[name] = {
                    "label_change_fraction": float(np.mean(y_new != ys)),
                    "coef_shift_std": float(np.std(w)),
                }
            else:
                base_coef = info.get("coef", None)
                if isinstance(base_coef, np.ndarray) and base_coef.shape[0] == Xs.shape[1]:
                    coef = base_coef + rng.normal(0.0, strength, size=base_coef.shape)
                    y_new = Xs @ coef
                else:
                    w = rng.normal(0.0, 1.0, size=Xs.shape[1])
                    y_new = ys + strength * (Xs @ w)

                sigma = float(info.get("noise_std", 0.0) or 0.0)
                if sigma > 0:
                    y_new = y_new + rng.normal(0.0, sigma, size=Xs.shape[0])
                split_data[name]["y"] = y_new
                details[name] = {
                    "target_std": float(np.std(y_new)),
                    "delta_std": float(np.std(y_new - ys)),
                }

    else:
        raise ValueError("domain_shift debe ser 'none', 'covariate', 'prior' o 'concept'")

    return {
        "mode": mode,
        "strength": strength,
        "shifted_splits": targets,
        "details": details,
    }


def _instance_dependent_flip_probs(
    X: Array,
    y: Array,
    rate: float,
    strength: float,
    n_classes: int,
) -> Array:
    scores = _estimate_class_scores(X, y, n_classes)
    true_score = scores[np.arange(y.size), y.astype(int)]

    scores_masked = scores.copy()
    scores_masked[np.arange(y.size), y.astype(int)] = -np.inf
    alt_score = np.max(scores_masked, axis=1)

    margin = true_score - alt_score
    margin = (margin - np.mean(margin)) / (np.std(margin) + 1e-12)
    hard = 1.0 / (1.0 + np.exp(margin * (1.0 + abs(strength))))
    p = rate * (0.5 + hard)
    return np.clip(p, 0.0, 1.0)


def _apply_label_noise(
    X: Array,
    y: Array,
    info: Dict[str, Any],
    rng: np.random.Generator,
    label_noise: str,
    label_noise_rate: float,
    label_noise_strength: float,
    batch_id: Optional[Array] = None,
) -> Tuple[Array, Dict[str, Any]]:
    mode = str(label_noise).lower()
    rate = float(np.clip(label_noise_rate, 0.0, 1.0))
    strength = float(max(0.0, label_noise_strength))

    if mode == "none" or rate == 0.0:
        return y, {"enabled": False, "mode": "none", "rate": rate}

    y_clean = y.copy()
    y_noisy = y.copy()

    if info.get("task") == "classification":
        y_int = y_clean.astype(int)
        k = int(info.get("n_classes", np.max(y_int) + 1))
        flip_mask = np.zeros(y_int.shape[0], dtype=bool)

        if mode == "random":
            flip_mask = rng.random(y_int.shape[0]) < rate

        elif mode == "batch_dependent":
            if batch_id is None:
                flip_mask = rng.random(y_int.shape[0]) < rate
                batch_rates = None
            else:
                b = int(np.max(batch_id)) + 1 if batch_id.size > 0 else 0
                if b == 0:
                    flip_mask = rng.random(y_int.shape[0]) < rate
                    batch_rates = None
                else:
                    jitter = rng.normal(0.0, max(0.1, strength + 0.1), size=b)
                    batch_rates = np.clip(rate * (1.0 + jitter), 0.0, 1.0)
                    flip_prob = batch_rates[batch_id.astype(int)]
                    flip_mask = rng.random(y_int.shape[0]) < flip_prob

        elif mode == "instance_dependent":
            flip_prob = _instance_dependent_flip_probs(X, y_int, rate, strength, k)
            flip_mask = rng.random(y_int.shape[0]) < flip_prob

        else:
            raise ValueError("label_noise debe ser 'none', 'random', 'batch_dependent' o 'instance_dependent'")

        if np.any(flip_mask):
            # Adversarial: choose least likely class proxy by centroid scores.
            scores = _estimate_class_scores(X, y_int, k)
            for i in np.flatnonzero(flip_mask):
                current = y_int[i]
                row = scores[i].copy()
                row[current] = np.inf
                target = int(np.argmin(row))
                if target == current:
                    target = int((current + 1) % k)
                y_noisy[i] = target

        meta = {
            "enabled": True,
            "mode": mode,
            "rate": rate,
            "strength": strength,
            "flipped_fraction": float(np.mean(y_noisy != y_clean)),
            "clean_y": y_clean,
        }
        if mode == "batch_dependent" and batch_id is not None:
            b = int(np.max(batch_id)) + 1 if batch_id.size > 0 else 0
            if b > 0:
                observed = []
                for bi in range(b):
                    m = batch_id == bi
                    if np.any(m):
                        observed.append(float(np.mean(y_noisy[m] != y_clean[m])))
                    else:
                        observed.append(0.0)
                meta["observed_batch_flip_rates"] = np.asarray(observed, dtype=float)
        return y_noisy.astype(y.dtype), meta

    # Regression noise model.
    if mode == "random":
        m = rng.random(y.shape[0]) < rate
        scale = max(float(np.std(y)), 1e-8)
        delta = rng.normal(0.0, scale * max(1.0, strength), size=y.shape[0])
        y_noisy[m] = y_noisy[m] + delta[m]

    elif mode == "batch_dependent":
        scale = max(float(np.std(y)), 1e-8)
        if batch_id is None:
            m = rng.random(y.shape[0]) < rate
            delta = rng.normal(0.0, scale * max(1.0, strength), size=y.shape[0])
            y_noisy[m] = y_noisy[m] + delta[m]
        else:
            b = int(np.max(batch_id)) + 1 if batch_id.size > 0 else 0
            jitter = rng.normal(0.0, max(0.1, strength + 0.1), size=max(1, b))
            br = np.clip(rate * (1.0 + jitter), 0.0, 1.0)
            probs = br[batch_id.astype(int)] if b > 0 else np.full(y.shape[0], rate)
            m = rng.random(y.shape[0]) < probs
            delta = rng.normal(0.0, scale * max(1.0, strength), size=y.shape[0])
            y_noisy[m] = y_noisy[m] + delta[m]

    elif mode == "instance_dependent":
        z = np.abs((y - np.mean(y)) / (np.std(y) + 1e-12))
        p = np.clip(rate * (1.0 + strength * z), 0.0, 1.0)
        m = rng.random(y.shape[0]) < p
        scale = max(float(np.std(y)), 1e-8)
        heavy = rng.standard_t(df=3, size=y.shape[0])
        y_noisy[m] = y_noisy[m] + heavy[m] * scale

    else:
        raise ValueError("label_noise debe ser 'none', 'random', 'batch_dependent' o 'instance_dependent'")

    return y_noisy.astype(y.dtype), {
        "enabled": True,
        "mode": mode,
        "rate": rate,
        "strength": strength,
        "perturbed_fraction": float(np.mean(y_noisy != y_clean)),
        "clean_y": y_clean,
    }


def _top_singular_vector(X: Array) -> Array:
    if X.size == 0:
        return np.array([], dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return vt[0]


def _inject_outliers(
    X: Array,
    y: Array,
    info: Dict[str, Any],
    rng: np.random.Generator,
    outliers: str,
    outlier_rate: float,
    outlier_magnitude: float,
    outlier_mode: str,
    outlier_heavy_tail: bool,
) -> Tuple[Array, Array, Dict[str, Any]]:
    typ = str(outliers).lower()
    rate = float(np.clip(outlier_rate, 0.0, 1.0))
    mag = float(max(0.0, outlier_magnitude))
    mode = str(outlier_mode).lower()

    if typ == "none" or rate == 0.0 or mag == 0.0:
        return X, y, {"enabled": False, "type": "none", "rate": rate, "magnitude": mag}

    if mode not in ("additive", "replace", "leverage"):
        raise ValueError("outlier_mode debe ser 'additive', 'replace' o 'leverage'")

    n, p = X.shape
    n_out = int(round(rate * n))
    n_out = int(np.clip(n_out, 0, n))
    if n_out == 0:
        return X, y, {"enabled": False, "type": typ, "rate": rate, "magnitude": mag}

    idx = rng.choice(n, size=n_out, replace=False)
    x_mask = np.zeros(n, dtype=bool)
    y_mask = np.zeros(n, dtype=bool)

    X2 = X.copy()
    y2 = y.copy()

    noise_draw = rng.standard_t(df=2, size=(n_out,)) if outlier_heavy_tail else rng.normal(0.0, 1.0, size=(n_out,))

    if typ in ("feature", "both") and p > 0:
        x_mask[idx] = True
        k_feat = max(1, int(round(0.1 * p)))
        feat_idx = np.sort(rng.choice(p, size=k_feat, replace=False))

        if mode == "leverage":
            v = _top_singular_vector(X2)
            if v.size == 0:
                v = np.zeros(p)
            shift = np.outer(mag * noise_draw, v)
            X2[idx] = X2[idx] + shift
        else:
            direction = rng.normal(0.0, 1.0, size=(n_out, k_feat))
            norm = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
            direction = direction / norm
            delta = mag * noise_draw[:, None] * direction
            if mode == "additive":
                X2[np.ix_(idx, feat_idx)] = X2[np.ix_(idx, feat_idx)] + delta
            elif mode == "replace":
                X2[np.ix_(idx, feat_idx)] = delta

    if typ in ("label", "both"):
        y_mask[idx] = True
        if info.get("task") == "classification":
            y_int = y2.astype(int)
            k = int(info.get("n_classes", np.max(y_int) + 1))
            scores = _estimate_class_scores(X2, y_int, k)
            for i in idx:
                current = int(y_int[i])
                row = scores[i].copy()
                row[current] = np.inf
                target = int(np.argmin(row))
                if target == current:
                    target = int((current + 1) % k)
                y2[i] = target
        else:
            scale = max(float(np.std(y2)), 1e-8)
            perturb = mag * scale * noise_draw
            y2[idx] = y2[idx] + perturb

    meta = {
        "enabled": True,
        "type": typ,
        "rate": rate,
        "magnitude": mag,
        "mode": mode,
        "heavy_tail": bool(outlier_heavy_tail),
        "sample_outlier_fraction": float(n_out / max(1, n)),
        "feature_outlier_mask": x_mask,
        "label_outlier_mask": y_mask,
        "outlier_indices": idx,
    }
    return X2, y2, meta


def _to_csv_matrix(path: Path, X: Array, prefix: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    X2 = np.asarray(X)
    p = X2.shape[1]
    header = ",".join([f"{prefix}{j}" for j in range(p)])
    if np.all(np.isfinite(X2)) and np.allclose(X2, np.rint(X2)):
        np.savetxt(path, np.rint(X2).astype(np.int64), delimiter=",", header=header, comments="", fmt="%d")
    else:
        np.savetxt(path, X2, delimiter=",", header=header, comments="", fmt="%.10g")


def _to_csv_vector(path: Path, y: Array, name: str = "y") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y2 = np.asarray(y).reshape(-1, 1)
    if np.all(np.isfinite(y2)) and np.allclose(y2, np.rint(y2)):
        np.savetxt(path, np.rint(y2).astype(np.int64), delimiter=",", header=name, comments="", fmt="%d")
    else:
        np.savetxt(path, y2, delimiter=",", header=name, comments="", fmt="%.10g")


def _save_dataset_folder(
    dataset_name: str,
    X: Optional[Array],
    y: Optional[Array],
    info: Dict[str, Any],
    X_train: Optional[Array] = None,
    y_train: Optional[Array] = None,
    X_val: Optional[Array] = None,
    y_val: Optional[Array] = None,
    X_test: Optional[Array] = None,
    y_test: Optional[Array] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    root = Path(__file__).resolve().parent
    # By default datasets are written under DATA/. A custom base_dir can
    # redirect outputs to another folder (for example, the script folder).
    out_root = (base_dir if base_dir is not None else (root / "DATA"))
    out = out_root / dataset_name
    # Recreate target folder from scratch to avoid stale files.
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=False)

    if X is not None and y is not None:
        _to_csv_matrix(out / "X.csv", X)
        _to_csv_vector(out / "Y.csv", y)
        # Ensure legacy split files do not remain from previous runs in the same folder.
        for fname in ("X_train.csv", "Y_train.csv", "X_val.csv", "Y_val.csv", "X_test.csv", "Y_test.csv"):
            p = out / fname
            if p.exists():
                p.unlink()
    elif X_train is not None and y_train is not None and X_val is not None and y_val is not None and X_test is not None and y_test is not None:
        X_all = np.vstack([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        _to_csv_matrix(out / "X.csv", X_all)
        _to_csv_vector(out / "Y.csv", y_all)
        _to_csv_matrix(out / "X_train.csv", X_train)
        _to_csv_vector(out / "Y_train.csv", y_train)
        _to_csv_matrix(out / "X_val.csv", X_val)
        _to_csv_vector(out / "Y_val.csv", y_val)
        _to_csv_matrix(out / "X_test.csv", X_test)
        _to_csv_vector(out / "Y_test.csv", y_test)

    info_path = out / "info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonify_info(info), f, ensure_ascii=False, indent=2)

    return out


def _jsonify_info(info: Dict[str, Any]) -> Dict[str, Any]:
    def conv(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            if v.size > 5000:
                return {
                    "__ndarray_summary__": True,
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "mean": float(np.mean(v)) if np.issubdtype(v.dtype, np.number) else None,
                    "std": float(np.std(v)) if np.issubdtype(v.dtype, np.number) else None,
                }
            return v.tolist()
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, dict):
            return {str(k): conv(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v

    return {str(k): conv(v) for k, v in info.items()}


def _mutual_info_from_counts(counts: Array) -> float:
    n = counts.sum()
    if n <= 0:
        return 0.0
    pxy = counts / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pxy > 0, pxy / np.maximum(px @ py, 1e-12), 1.0)
        mi = np.where(pxy > 0, pxy * np.log(ratio), 0.0)
    return float(np.sum(mi))


def _cramers_v(counts: Array) -> float:
    n = counts.sum()
    if n <= 0:
        return 0.0
    row = counts.sum(axis=1, keepdims=True)
    col = counts.sum(axis=0, keepdims=True)
    expected = row @ col / max(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(expected > 0, (counts - expected) ** 2 / expected, 0.0).sum()
    r, c = counts.shape
    denom = max(1, min(r - 1, c - 1))
    return float(np.sqrt((chi2 / max(n, 1)) / denom))


def summarize_dataset(X: Optional[Array], y: Optional[Array], info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summary stats for generated datasets.

    Includes class balance (global/split), batch-label association (Cramér's V and MI),
    split mean/variance shifts and outlier confirmation rates.
    """
    out: Dict[str, Any] = {
        "model": info.get("model"),
        "task": info.get("task"),
    }

    def _class_balance(arr: Array) -> Dict[str, Any]:
        if arr.size == 0:
            return {"counts": [], "probs": []}
        vals, counts = np.unique(arr, return_counts=True)
        probs = counts / counts.sum()
        return {
            "labels": vals.astype(int) if np.issubdtype(vals.dtype, np.integer) else vals,
            "counts": counts.astype(int),
            "probs": probs.astype(float),
        }

    split_info = info.get("splits", None)
    if isinstance(split_info, dict) and split_info:
        per_split = {}
        for key in ("train", "val", "test"):
            yk = split_info.get(key, {}).get("y")
            if yk is not None:
                per_split[key] = {
                    "n": int(len(yk)),
                    "class_balance": _class_balance(np.asarray(yk)),
                }
                xk = split_info.get(key, {}).get("feature_mean")
                vk = split_info.get(key, {}).get("feature_var")
                if xk is not None:
                    per_split[key]["feature_mean"] = float(xk)
                if vk is not None:
                    per_split[key]["feature_var"] = float(vk)
        out["split_stats"] = per_split

        for name in ("val", "test"):
            key = f"shift_train_vs_{name}"
            if key in split_info:
                out[key] = split_info[key]
    elif y is not None:
        out["n"] = int(len(y))
        out["class_balance"] = _class_balance(np.asarray(y)) if info.get("task") == "classification" else None
        if X is not None:
            out["feature_mean"] = float(np.mean(X))
            out["feature_var"] = float(np.var(X))

    batch_id = info.get("batch_id")
    clean_y = info.get("clean_y")
    if batch_id is not None:
        y_ref = clean_y if clean_y is not None else y
        if y_ref is not None:
            b = np.asarray(batch_id).astype(int)
            yr = np.asarray(y_ref).astype(int)
            n_b = int(np.max(b)) + 1 if b.size > 0 else 0
            n_y = int(np.max(yr)) + 1 if yr.size > 0 else 0
            if n_b > 0 and n_y > 0:
                counts = np.zeros((n_b, n_y), dtype=float)
                for bi, yi in zip(b, yr):
                    counts[bi, yi] += 1.0
                out["batch_label_association"] = {
                    "cramers_v": _cramers_v(counts),
                    "mutual_info": _mutual_info_from_counts(counts),
                }

    out_meta = info.get("outlier_params", {})
    if isinstance(out_meta, dict) and out_meta.get("enabled", False):
        feat_mask = out_meta.get("feature_outlier_mask", None)
        lab_mask = out_meta.get("label_outlier_mask", None)
        out["outlier_fraction"] = {
            "feature": float(np.mean(feat_mask)) if isinstance(feat_mask, np.ndarray) else None,
            "label": float(np.mean(lab_mask)) if isinstance(lab_mask, np.ndarray) else None,
        }

    return _jsonify_info(out)


def make_benchmark_dataset(
    n_samples: int = 1000,
    n_features: int = 100,
    n_informative: int = 10,
    # Tarea y modelo
    task: str = "classification",  # Se infiere a partir de 'model'; aquí es informativo
    model: str = "sparse_logistic",  # 'sparse_logistic', 'sparse_linear', 'friedman1', 'xor', 'parity', 'madelon_like', 'mirna_nb'
    # Estructura de X (gaussiano)
    corr: str = "independent",  # 'independent', 'toeplitz', 'block'
    rho: float = 0.3,  # intensidad de correlación (Toeplitz/Block)
    n_blocks: int = 5,  # nº de bloques si corr='block'
    # Señal/ruido y efectos
    snr: float = 5.0,  # SNR para regresión; cuántos "signal variances" por "noise variance"
    effect_size: float = 1.0,  # magnitud de coeficientes (lineal) o log-fold-change (miRNA)
    heteroscedastic: float = 0.0,  # 0 => homoscedástico; >0 => var(noise) crece con |Xβ|
    # Clasificación
    n_classes: int = 2,
    class_balance: float = 0.5,  # mantenido por compatibilidad (binario)
    class_probs: Optional[Sequence[float]] = None,  # distribución objetivo de clases
    # Redundantes/ruido extra
    n_redundant: int = 0,
    n_repeated: int = 0,
    n_categorical: int = 0,
    n_categories: int = 3,
    # Opcional: fijar soporte
    support_mask: Optional[np.ndarray] = None,
    # Parámetros específicos miRNA
    mirna_library_log_mean: float = 11.0,
    mirna_library_log_sd: float = 0.5,
    mirna_mean_log_expr: float = 1.5,
    mirna_sd_log_expr: float = 1.0,
    mirna_dispersion_scale: float = 1.0,
    dropout_rate: float = 0.0,
    n_modules: int = 0,
    module_sd: float = 0.3,
    random_state: Optional[int] = None,
    # Nuevos parámetros: batches/confounders/shifts
    n_batches: int = 0,
    batch_probs: Optional[Sequence[float]] = None,
    batch_feature_shift: float = 0.0,
    batch_feature_scale: float = 0.0,
    batch_effect_on_subset: float = 1.0,
    batch_confounded_with_label: float = 0.0,
    latent_confounders: int = 0,
    latent_strength: float = 0.0,
    latent_link_to_label: float = 0.0,
    domain_shift: str = "none",
    domain_shift_strength: float = 0.0,
    return_splits: bool = False,
    split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    shift_applies_to: str = "test",
    # Nuevos parámetros: label noise
    label_noise: str = "none",
    label_noise_rate: float = 0.0,
    label_noise_strength: float = 0.0,
    # Nuevos parámetros: outliers
    outliers: str = "none",
    outlier_rate: float = 0.0,
    outlier_magnitude: float = 0.0,
    outlier_mode: str = "additive",
    outlier_heavy_tail: bool = False,
) -> Union[Tuple[Array, Array, Dict[str, Any]], Dict[str, Any]]:
    """
    Crea un dataset sintético con soporte conocido de variables relevantes.

    Backward compatibility:
    - Con los nuevos parámetros en sus defaults, mantiene el comportamiento previo
      (retorno legacy `(X, y, info)` y sin perturbaciones nuevas).

    Nuevos parámetros relevantes:
    - `n_batches`, `batch_*`: efectos de batch sobre X y confounding batch-label.
    - `latent_*`: confusores latentes U que influyen en X y opcionalmente en y.
    - `domain_shift`, `return_splits`, `split`, `shift_applies_to`: generación
      train/val/test con shifts en val/test.
    - `label_noise*`: ruido de etiquetas realista.
    - `outliers*`: inyección de outliers en features y/o labels.

    Retorna:
    - `return_splits=False`: `(X, y, info)`.
    - `return_splits=True`: dict con `X_train/y_train/X_val/y_val/X_test/y_test/info`.
    """
    rng = _rng(random_state)

    # ====== Preparación de probabilidades de clase ======
    if class_probs is None:
        if n_classes == 2:
            class_probs = [1 - class_balance, class_balance]
        else:
            class_probs = [1.0 / n_classes] * n_classes
    class_probs = np.asarray(class_probs, dtype=float)
    if len(class_probs) != n_classes:
        raise ValueError("class_probs debe tener longitud n_classes")
    class_probs = class_probs / class_probs.sum()

    eta = None
    eta_no_b = None

    # ====== MODELOS ESPECÍFICOS ======
    if model in ("sparse_logistic", "sparse_linear"):
        X = _sample_gaussian_features(n_samples, n_features, corr=corr, rho=rho, n_blocks=n_blocks, rng=rng)
        support = _build_support(n_features, n_informative, rng, support_mask)

        if model == "sparse_linear":
            coef = np.zeros(n_features)
            magnitudes = rng.uniform(0.5, 1.5, size=support.sum()) * effect_size
            signs = rng.choice([-1.0, 1.0], size=support.sum())
            coef[support] = magnitudes * signs
            eta = X @ coef
            sigma = _scale_noise_for_snr(eta, snr=snr)
            if heteroscedastic > 0:
                noise = rng.normal(0, sigma, size=n_samples) * (1.0 + heteroscedastic * np.abs(eta))
            else:
                noise = rng.normal(0, sigma, size=n_samples)
            y = eta + noise
            base_noise_std = float(np.std(eta))

            info = {
                "model": model,
                "task": "regression",
                "support": support,
                "support_with_redundant": support.copy(),
                "coef": coef,
                "coef_matrix": None,
                "snr": snr,
                "noise_std": base_noise_std,
                "corr": corr,
                "rho": rho,
                "n_blocks": n_blocks,
                "n_classes": None,
                "class_probs": None,
                "notes": "Diseño gaussiano con soporte escaso; coeficientes aleatorios (regresión).",
            }

        else:
            K = int(n_classes)
            coef_matrix = np.zeros((n_features, K))
            magnitudes = rng.uniform(0.5, 1.5, size=(support.sum(), K)) * effect_size
            signs = rng.choice([-1.0, 1.0], size=(support.sum(), K))
            coef_matrix[support, :] = magnitudes * signs
            eta_no_b = X @ coef_matrix
            b = _calibrate_softmax_intercepts(eta_no_b, class_probs)
            eta = eta_no_b + b
            P = _softmax(eta)
            cumP = np.cumsum(P, axis=1)
            r = rng.random(n_samples)[:, None]
            y = (cumP < r).sum(axis=1)
            info = {
                "model": model,
                "task": "classification",
                "support": support,
                "support_with_redundant": support.copy(),
                "coef": None,
                "coef_matrix": coef_matrix,
                "snr": None,
                "noise_std": float(np.std(eta_no_b)),
                "corr": corr,
                "rho": rho,
                "n_blocks": n_blocks,
                "n_classes": K,
                "class_probs": class_probs,
                "notes": "Clasificación softmax con soporte escaso y X gaussiano.",
            }

    elif model == "friedman1":
        p0 = max(n_features, 10)
        X = rng.uniform(0.0, 1.0, size=(n_samples, p0))
        y_signal = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
        sigma = _scale_noise_for_snr(y_signal, snr=snr)
        y = y_signal + rng.normal(0, sigma, size=n_samples)
        support = np.zeros(p0, dtype=bool)
        support[:5] = True
        info = {
            "model": "friedman1",
            "task": "regression",
            "support": support[:n_features],
            "support_with_redundant": support[:n_features].copy(),
            "coef": None,
            "coef_matrix": None,
            "snr": snr,
            "noise_std": sigma,
            "corr": "independent",
            "rho": 0.0,
            "n_blocks": None,
            "n_classes": None,
            "class_probs": None,
            "notes": "Friedman #1: no lineal aditivo con 5 variables relevantes (x1..x5).",
        }
        if p0 != n_features:
            if n_features < p0:
                X = X[:, :n_features]
            else:
                extra = rng.uniform(0.0, 1.0, size=(n_samples, n_features - p0))
                X = np.concatenate([X, extra], axis=1)

    elif model in ("xor", "parity"):
        k = int(max(2, n_informative))
        bits = rng.integers(0, 2, size=(n_samples, k))
        jitter = rng.normal(0, 0.1, size=(n_samples, k))
        informative = bits.astype(float) + jitter
        p_noise = max(0, n_features - k)
        noise = rng.standard_normal((n_samples, p_noise)) if p_noise > 0 else None
        X = np.concatenate([informative, noise], axis=1) if noise is not None else informative
        K = int(n_classes)
        y = (bits.sum(axis=1) % K).astype(int)
        support = np.zeros(X.shape[1], dtype=bool)
        support[:k] = True
        info = {
            "model": model,
            "task": "classification",
            "support": support,
            "support_with_redundant": support.copy(),
            "coef": None,
            "coef_matrix": None,
            "snr": None,
            "noise_std": 0.1,
            "corr": "independent",
            "rho": 0.0,
            "n_blocks": None,
            "n_classes": K,
            "class_probs": class_probs,
            "notes": "Etiqueta = (paridad/XOR de k bits) mod K, con jitter.",
        }

    elif model == "madelon_like":
        base_dim = 5
        vertices = np.array(np.meshgrid(*[[-1, 1]] * base_dim)).T.reshape(-1, base_dim)
        K = int(n_classes)
        labels = np.repeat(np.arange(K), repeats=len(vertices) // K)
        if labels.size < len(vertices):
            labels = np.r_[labels, np.arange(len(vertices) - labels.size) % K]
        labels = labels[: len(vertices)]
        labels = rng.permutation(labels)

        centers_idx = rng.integers(0, len(vertices), size=n_samples)
        centers = vertices[centers_idx]
        y = labels[centers_idx]
        sep = 2.0
        base = centers * sep + rng.normal(0, 1.0, size=(n_samples, base_dim))

        n_lin = min(15, max(0, n_features - base_dim))
        lin_cols = []
        red_map = []
        for j in range(n_lin):
            w = rng.normal(0, 1, size=base_dim)
            col = base @ w + rng.normal(0, 0.2, size=n_samples)
            lin_cols.append(col)
        if lin_cols:
            lin_cols = np.column_stack(lin_cols)
            X = np.column_stack([base, lin_cols])
            red_map = [(base_dim + j, list(range(base_dim))) for j in range(n_lin)]
        else:
            X = base
        if X.shape[1] < n_features:
            X = np.column_stack([X, rng.standard_normal((n_samples, n_features - X.shape[1]))])
        support = np.zeros(X.shape[1], dtype=bool)
        support[:base_dim] = True
        support_with_redundant = support.copy()
        support_with_redundant[base_dim : base_dim + n_lin] = True
        info = {
            "model": "madelon_like",
            "task": "classification",
            "support": support,
            "support_with_redundant": support_with_redundant,
            "coef": None,
            "coef_matrix": None,
            "snr": None,
            "noise_std": 1.0,
            "corr": "independent",
            "rho": 0.0,
            "n_blocks": None,
            "redundant_map": red_map,
            "n_classes": K,
            "class_probs": class_probs,
            "notes": f"Clústeres en hipercubo 5D; 5 informativas + {n_lin} combinaciones lineales + ruido, con {K} clases.",
        }

    elif model == "mirna_nb":
        K = int(n_classes)
        cum = np.cumsum(class_probs)
        r = rng.random(n_samples)
        y = np.searchsorted(cum, r).astype(int)

        p = n_features
        pi_g = _sample_nb_means(p, rng, mean_log_expr=mirna_mean_log_expr, sd_log_expr=mirna_sd_log_expr)
        L = _sample_library_sizes(n_samples, rng, log_mean=mirna_library_log_mean, log_sd=mirna_library_log_sd)

        module_of = None
        sample_module_effect = None
        if n_modules and n_modules > 0:
            module_of = rng.integers(0, n_modules, size=p)
            sample_module_effect = rng.normal(0.0, module_sd, size=(n_samples, n_modules))

        support = _build_support(p, n_informative, rng, support_mask)

        de_log_fc = np.zeros((p, K), dtype=float)
        for g in np.flatnonzero(support):
            de_log_fc[g, 1:] = rng.normal(0.0, effect_size, size=K - 1)

        log_mu = np.log(L)[:, None] + np.log(pi_g)[None, :]
        if n_modules and n_modules > 0:
            m_idx = module_of[None, :]
            mod_eff = np.take_along_axis(sample_module_effect, m_idx, axis=1)
            log_mu = log_mu + mod_eff

        class_shift_per_gene = de_log_fc[:, y].T
        log_mu = log_mu + class_shift_per_gene
        mu = np.exp(log_mu)

        alpha_g = np.exp(rng.normal(-1.5, 0.5, size=p)) * float(mirna_dispersion_scale)
        X = _sample_nb_counts(mu, alpha_g, rng)

        if dropout_rate and dropout_rate > 0.0:
            mask = rng.random(size=X.shape) < float(dropout_rate)
            X[mask] = 0

        info = {
            "model": "mirna_nb",
            "task": "classification",
            "support": support,
            "support_with_redundant": support.copy(),
            "coef": None,
            "coef_matrix": None,
            "snr": None,
            "noise_std": None,
            "corr": None,
            "rho": None,
            "n_blocks": None,
            "n_classes": K,
            "class_probs": class_probs,
            "is_counts": True,
            "library_sizes": L,
            "base_proportions": pi_g,
            "dispersion": alpha_g,
            "de_log_fc": de_log_fc,
            "module_of": module_of,
            "module_sd": module_sd,
            "notes": "Conteos NB por miRNA con tamaños de biblioteca, DE multiclase y módulos de coexpresión opcionales.",
        }

    else:
        raise ValueError(f"model '{model}' no soportado.")

    # ====== OPCIONALES PREEXISTENTES ======
    redundant_map = list(info.get("redundant_map", []))
    repeated_from = []

    if model not in ("madelon_like", "friedman1", "mirna_nb"):
        support_idx = np.flatnonzero(info["support"])
        if n_redundant > 0:
            X, red_map_extra = _add_redundant_features(X, support_idx, n_redundant, noise_std=0.1, rng=rng)
            redundant_map.extend(red_map_extra)
            supp = info["support_with_redundant"]
            if supp.shape[0] < X.shape[1]:
                supp = np.r_[supp, np.zeros(X.shape[1] - supp.shape[0], dtype=bool)]
            for j, _ in red_map_extra:
                supp[j] = True
            info["support_with_redundant"] = supp

    if model not in ("mirna_nb",):
        if n_repeated > 0:
            X, rep = _add_repeated_features(X, n_repeated, rng)
            repeated_from.extend(rep)

        if n_categorical > 0:
            X, categorical_idx = _add_categorical_noise(X, n_categorical, n_categories, rng)
        else:
            categorical_idx = info.get("categorical_idx", [])
    else:
        categorical_idx = info.get("categorical_idx", [])

    if X.shape[1] < n_features and model not in ("mirna_nb",):
        X = np.column_stack([X, rng.standard_normal((n_samples, n_features - X.shape[1]))])

    # ====== NUEVO: batches ======
    batch_id, batch_probs_used, batch_meta = _assign_batches(
        n_samples=n_samples,
        n_batches=int(n_batches),
        rng=rng,
        batch_probs=batch_probs,
        y=y,
        task=info.get("task", task),
        batch_confounded_with_label=batch_confounded_with_label,
        n_classes=int(info.get("n_classes") or n_classes),
    )
    if batch_id is not None:
        X, batch_effect_meta = _apply_batch_effects(
            X=X,
            batch_id=batch_id,
            n_batches=int(n_batches),
            batch_feature_shift=batch_feature_shift,
            batch_feature_scale=batch_feature_scale,
            batch_effect_on_subset=batch_effect_on_subset,
            rng=rng,
        )
    else:
        batch_effect_meta = {"enabled": False}

    # ====== NUEVO: confusores latentes ======
    X, y, latent_meta = _apply_latent_confounders(
        X=X,
        y=y,
        info=info,
        latent_confounders=latent_confounders,
        latent_strength=latent_strength,
        latent_link_to_label=latent_link_to_label,
        rng=rng,
    )

    # Actualizamos info básico antes de ruido/outliers/splits.
    p_final = X.shape[1]
    for key in ("support", "support_with_redundant"):
        mask = info.get(key, None)
        if mask is None:
            continue
        if mask.shape[0] < p_final:
            pad = np.zeros(p_final - mask.shape[0], dtype=bool)
            mask = np.r_[mask, pad]
        else:
            mask = mask[:p_final]
        info[key] = mask

    info["redundant_map"] = redundant_map
    info["repeated_from"] = repeated_from
    info["categorical_idx"] = categorical_idx
    info["n_features_final"] = p_final

    info["batch_id"] = batch_id
    info["batch_params"] = {
        **batch_meta,
        "batch_probs": batch_probs_used,
        "batch_effect": batch_effect_meta,
    }
    info["latent_params"] = latent_meta

    notes_extra = []
    if int(n_batches) > 0:
        notes_extra.append("batch effects enabled")
    if int(latent_confounders) > 0 and (latent_strength != 0.0 or latent_link_to_label != 0.0):
        notes_extra.append("latent confounders enabled")
    if label_noise != "none" and label_noise_rate > 0:
        notes_extra.append(f"label noise ({label_noise})")
    if outliers != "none" and outlier_rate > 0 and outlier_magnitude > 0:
        notes_extra.append(f"outliers ({outliers})")
    if return_splits:
        notes_extra.append(f"splits + domain shift ({domain_shift})")
    if notes_extra:
        info["notes"] = f"{info.get('notes', '')} | " + "; ".join(notes_extra)
    info["clean_y"] = y.copy()

    # ====== Camino legacy: sin splits ======
    if not return_splits:
        y_clean = y.copy()

        y_noisy, label_meta = _apply_label_noise(
            X=X,
            y=y,
            info=info,
            rng=rng,
            label_noise=label_noise,
            label_noise_rate=label_noise_rate,
            label_noise_strength=label_noise_strength,
            batch_id=batch_id,
        )
        X_out, y_out, outlier_meta = _inject_outliers(
            X=X,
            y=y_noisy,
            info=info,
            rng=rng,
            outliers=outliers,
            outlier_rate=outlier_rate,
            outlier_magnitude=outlier_magnitude,
            outlier_mode=outlier_mode,
            outlier_heavy_tail=outlier_heavy_tail,
        )

        info["clean_y"] = y_clean
        info["label_noise_params"] = label_meta
        info["outlier_params"] = outlier_meta
        info["domain_shift"] = {
            "mode": "none",
            "strength": float(domain_shift_strength),
            "shifted_splits": [],
            "details": {"note": "return_splits=False; domain_shift no aplicado"},
        }

        if info.get("is_counts", False):
            X_out = np.clip(np.rint(X_out), 0, None).astype(int)
            return X_out, y_out.astype(int) if info.get("task") == "classification" else y_out, info

        return X_out.astype(float), y_out, info

    # ====== Camino con splits + domain shift ======
    idx = _split_indices(n_samples, split, rng)
    split_data = {
        "train": {
            "X": X[idx["train"]].copy(),
            "y": y[idx["train"]].copy(),
            "batch_id": batch_id[idx["train"]].copy() if batch_id is not None else None,
        },
        "val": {
            "X": X[idx["val"]].copy(),
            "y": y[idx["val"]].copy(),
            "batch_id": batch_id[idx["val"]].copy() if batch_id is not None else None,
        },
        "test": {
            "X": X[idx["test"]].copy(),
            "y": y[idx["test"]].copy(),
            "batch_id": batch_id[idx["test"]].copy() if batch_id is not None else None,
        },
    }

    for k in ("train", "val", "test"):
        split_data[k]["clean_y"] = split_data[k]["y"].copy()

    domain_meta = _apply_domain_shift(
        split_data=split_data,
        info=info,
        domain_shift=domain_shift,
        domain_shift_strength=domain_shift_strength,
        shift_applies_to=shift_applies_to,
        rng=rng,
    )

    split_label_noise_params: Dict[str, Any] = {}
    split_outlier_params: Dict[str, Any] = {}
    for k in ("train", "val", "test"):
        y_noisy, label_meta = _apply_label_noise(
            X=split_data[k]["X"],
            y=split_data[k]["y"],
            info=info,
            rng=rng,
            label_noise=label_noise,
            label_noise_rate=label_noise_rate,
            label_noise_strength=label_noise_strength,
            batch_id=split_data[k].get("batch_id"),
        )
        X_out, y_out, outlier_meta = _inject_outliers(
            X=split_data[k]["X"],
            y=y_noisy,
            info=info,
            rng=rng,
            outliers=outliers,
            outlier_rate=outlier_rate,
            outlier_magnitude=outlier_magnitude,
            outlier_mode=outlier_mode,
            outlier_heavy_tail=outlier_heavy_tail,
        )
        split_data[k]["X"] = X_out
        split_data[k]["y"] = y_out
        split_data[k]["label_noise_params"] = label_meta
        split_data[k]["outlier_params"] = outlier_meta
        split_label_noise_params[k] = label_meta
        split_outlier_params[k] = outlier_meta

    info["domain_shift"] = domain_meta
    info["split"] = tuple(float(v) for v in _validate_split(split))
    info["shift_applies_to"] = shift_applies_to
    info["label_noise_params"] = {
        "mode": label_noise,
        "rate": float(label_noise_rate),
        "strength": float(label_noise_strength),
    }
    info["outlier_params"] = {
        "type": outliers,
        "rate": float(outlier_rate),
        "magnitude": float(outlier_magnitude),
        "mode": outlier_mode,
        "heavy_tail": bool(outlier_heavy_tail),
    }
    info["split_label_noise_params"] = split_label_noise_params
    info["split_outlier_params"] = split_outlier_params

    # Keep lightweight split summaries and labels in info (avoid storing full X arrays).
    info["splits"] = {
        key: {
            "y": split_data[key]["y"],
            "batch_id": split_data[key]["batch_id"],
            "clean_y": split_data[key]["clean_y"],
            "feature_mean": float(np.mean(split_data[key]["X"])),
            "feature_var": float(np.var(split_data[key]["X"])),
        }
        for key in ("train", "val", "test")
    }
    info["splits"]["shift_train_vs_val"] = {
        "mean_l2": float(np.linalg.norm(split_data["train"]["X"].mean(axis=0) - split_data["val"]["X"].mean(axis=0))),
        "var_l2": float(np.linalg.norm(split_data["train"]["X"].var(axis=0) - split_data["val"]["X"].var(axis=0))),
    }
    info["splits"]["shift_train_vs_test"] = {
        "mean_l2": float(np.linalg.norm(split_data["train"]["X"].mean(axis=0) - split_data["test"]["X"].mean(axis=0))),
        "var_l2": float(np.linalg.norm(split_data["train"]["X"].var(axis=0) - split_data["test"]["X"].var(axis=0))),
    }

    if info.get("is_counts", False):
        for key in ("train", "val", "test"):
            split_data[key]["X"] = np.clip(np.rint(split_data[key]["X"]), 0, None).astype(int)
            if info.get("task") == "classification":
                split_data[key]["y"] = split_data[key]["y"].astype(int)

    return {
        "X_train": split_data["train"]["X"],
        "y_train": split_data["train"]["y"],
        "X_val": split_data["val"]["X"],
        "y_val": split_data["val"]["y"],
        "X_test": split_data["test"]["X"],
        "y_test": split_data["test"]["y"],
        "info": info,
    }


def make_robustness_suite(config: Dict[str, Any], seeds: List[int]) -> List[Dict[str, Any]]:
    """
    Build a lightweight realism/robustness suite and persist datasets under DATA/.

    Expected `config` keys (all optional):
    - base_params: dict passed to make_benchmark_dataset
    - name_prefix: str
    - batch_strengths: list[float]
    - domain_shifts: list[tuple[str, float]]
    - outlier_settings: list[tuple[str, float, float]]
    - n_batches: int (used when batch_strength > 0)
    - include_data: bool (include full arrays in returned list)
    - save_datasets: bool (default True)
    """
    base_params = dict(config.get("base_params", {}))
    prefix = str(config.get("name_prefix", "robust"))
    batch_strengths = list(config.get("batch_strengths", [0.0, 0.5]))
    domain_shifts = list(config.get("domain_shifts", [("none", 0.0), ("covariate", 0.4), ("prior", 0.4), ("concept", 0.4)]))
    outlier_settings = list(config.get("outlier_settings", [("none", 0.0, 0.0), ("feature", 0.02, 6.0), ("label", 0.02, 4.0), ("both", 0.03, 6.0)]))
    n_batches_on = int(config.get("n_batches", 4))
    include_data = bool(config.get("include_data", False))
    save_datasets = bool(config.get("save_datasets", True))

    results: List[Dict[str, Any]] = []

    for seed in seeds:
        for b_strength in batch_strengths:
            for shift_mode, shift_strength in domain_shifts:
                for out_type, out_rate, out_mag in outlier_settings:
                    params = dict(base_params)
                    params.update(
                        {
                            "random_state": int(seed),
                            "return_splits": True,
                            "domain_shift": str(shift_mode),
                            "domain_shift_strength": float(shift_strength),
                            "outliers": str(out_type),
                            "outlier_rate": float(out_rate),
                            "outlier_magnitude": float(out_mag),
                            "n_batches": n_batches_on if float(b_strength) > 0 else 0,
                            "batch_feature_shift": float(b_strength),
                            "batch_feature_scale": float(b_strength) * 0.5,
                        }
                    )

                    ds = make_benchmark_dataset(**params)
                    assert isinstance(ds, dict)

                    name = (
                        f"{prefix}_seed{seed}_b{b_strength:.2f}_"
                        f"{shift_mode}{float(shift_strength):.2f}_"
                        f"o{out_type}_{float(out_rate):.3f}_{float(out_mag):.2f}"
                    ).replace(" ", "_")

                    if save_datasets:
                        _save_dataset_folder(
                            dataset_name=name,
                            X=None,
                            y=None,
                            info=ds["info"],
                            X_train=ds["X_train"],
                            y_train=ds["y_train"],
                            X_val=ds["X_val"],
                            y_val=ds["y_val"],
                            X_test=ds["X_test"],
                            y_test=ds["y_test"],
                        )

                    summary = summarize_dataset(None, None, ds["info"])
                    item = {
                        "name": name,
                        "seed": int(seed),
                        "params": {
                            "batch_strength": float(b_strength),
                            "domain_shift": str(shift_mode),
                            "domain_shift_strength": float(shift_strength),
                            "outliers": str(out_type),
                            "outlier_rate": float(out_rate),
                            "outlier_magnitude": float(out_mag),
                        },
                        "summary": summary,
                    }
                    if include_data:
                        item["dataset"] = ds

                    results.append(item)

    return results


if __name__ == "__main__":
    # Common settings used for both generated datasets.
    base_params = dict(
        model="mirna_nb",
        n_features=500,
        n_informative=40,
        n_classes=3,
        class_probs=[0.50, 0.30, 0.20],
        return_splits=True,
        random_state=42,
        mirna_library_log_sd=0.6,
        mirna_mean_log_expr=1.3,
        mirna_dispersion_scale=1.2,
        dropout_rate=0.08,
        n_modules=12,
        module_sd=0.45,
        effect_size=0.9,
        n_batches=4,
        domain_shift="covariate",
    )

    # Dataset-specific overrides; only values that differ are listed.
    dataset_overrides = {
        "mirna_nb_01": dict(
            batch_feature_shift=0.5,
            batch_feature_scale=0.20,
            batch_effect_on_subset=0.30,
            batch_confounded_with_label=0.45,
            latent_confounders=3,
            latent_strength=0.35,
            latent_link_to_label=0.20,
            label_noise="batch_dependent",
            label_noise_rate=0.06,
            label_noise_strength=0.50,
            domain_shift_strength=0.45,
        ),
        "mirna_nb_02": dict(
            batch_feature_shift=0.4,
            batch_feature_scale=0.25,
            batch_effect_on_subset=0.25,
            batch_confounded_with_label=0.35,
            latent_confounders=3,
            latent_strength=0.25,
            latent_link_to_label=0.10,
            domain_shift_strength=0.25,
            outliers="both",
            outlier_rate=0.1,
            outlier_magnitude=6.0,
            outlier_mode="leverage",
            outlier_heavy_tail=True,
        ),
    }

    output_root = Path(__file__).resolve().parent

    for dataset_name, overrides in dataset_overrides.items():
        params = dict(base_params)
        params.update(overrides)

        # 1) Generate data and splits from configuration.
        generated = make_benchmark_dataset(**params)
        if not isinstance(generated, dict):
            raise RuntimeError("Expected split output (dict) from make_benchmark_dataset")

        # 2) Merge train/val/test so X.csv and Y.csv contain the full dataset.
        info = generated["info"]
        X_all = np.vstack([generated["X_train"], generated["X_val"], generated["X_test"]])
        y_all = np.concatenate([generated["y_train"], generated["y_val"], generated["y_test"]])

        # 3) Normalize export dtypes for stable CSV outputs.
        if np.all(np.isfinite(X_all)) and np.allclose(X_all, np.rint(X_all)):
            X_all = np.clip(np.rint(X_all), 0, None).astype(int)
        if info.get("task") == "classification":
            y_all = np.rint(np.asarray(y_all)).astype(int)

        # 4) Save each dataset as a dedicated folder directly inside Paper/.
        out_dir = _save_dataset_folder(
            dataset_name=dataset_name,
            X=X_all,
            y=y_all,
            info=info,
            base_dir=output_root,
        )
        summary = summarize_dataset(X_all, y_all, info)
        print(f"Dataset created: {dataset_name}")
        print(f"Saved in: {out_dir}")
        print(f"Shapes: X={X_all.shape}, y={y_all.shape}")
        print(f"Summary: {summary}")
