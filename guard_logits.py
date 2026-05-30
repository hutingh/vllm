import torch
import torch.nn.functional as F

def logmeanexp(t, dim):
    return torch.logsumexp(t, dim=dim) - torch.log(torch.tensor(t.shape[dim], device=t.device))

def js_div(log_p, log_q):
    # log_p/log_q: [..., h]
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(1e-30))
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)

def guard_logits(
    baseline_logits,
    candidate_logits,
    *,
    alpha=3.0,
    gamma=2.0,
    abs_js_p99=1e-3,
    abs_js_p999=5e-3,
    large_margin=0.5,
    huge_margin=1.0,
    max_large_margin_flip_rate=0.0,
    max_huge_margin_top5_drop_rate=0.0,
    min_top10_overlap=0.8,
    max_centered_rel_l2_p99=0.05,
):
    """
    Return True if candidate is OK, False if abnormal.
    baseline_logits/candidate_logits: [n, m, x, h]
    """

    assert baseline_logits.shape == candidate_logits.shape
    n, m, x, h = baseline_logits.shape

    # 1. 用 log-prob 做主判断，规避 logits 整体平移。
    b_lp = F.log_softmax(baseline_logits.float(), dim=-1)
    c_lp = F.log_softmax(candidate_logits.float(), dim=-1)

    # 2. 每条样本、每个 token 的“重复运行平均分布”。
    # shape: [n, x, h]
    b_mean_lp = logmeanexp(b_lp, dim=1)
    c_mean_lp = logmeanexp(c_lp, dim=1)

    # 3. baseline 自身噪声：每次 baseline run 到 baseline 平均分布的 JS。
    # shape: [n, m, x]
    b_noise = js_div(b_lp, b_mean_lp[:, None, :, :])

    # 4. candidate 自身噪声。
    c_noise = js_div(c_lp, c_mean_lp[:, None, :, :])

    # 5. candidate 相对 baseline 的中心漂移。
    # shape: [n, x]
    drift = js_div(b_mean_lp, c_mean_lp)

    b_noise_p99 = torch.quantile(b_noise.flatten(), 0.99)
    b_noise_p999 = torch.quantile(b_noise.flatten(), 0.999)

    c_noise_p99 = torch.quantile(c_noise.flatten(), 0.99)
    drift_p99 = torch.quantile(drift.flatten(), 0.99)
    drift_p999 = torch.quantile(drift.flatten(), 0.999)

    # 6. large-margin top1 flip。
    b_top = torch.topk(b_mean_lp, k=10, dim=-1)
    c_top = torch.topk(c_mean_lp, k=10, dim=-1)

    b_top1 = b_top.indices[..., 0]
    c_top1 = c_top.indices[..., 0]
    b_margin = b_top.values[..., 0] - b_top.values[..., 1]

    large_mask = b_margin > large_margin
    huge_mask = b_margin > huge_margin

    large_flip = (b_top1 != c_top1) & large_mask

    # huge margin 下，baseline top1 至少应该还在 candidate top5 内。
    c_top5 = c_top.indices[..., :5]
    b_top1_in_c_top5 = (c_top5 == b_top1[..., None]).any(dim=-1)
    huge_top5_drop = (~b_top1_in_c_top5) & huge_mask

    large_flip_rate = large_flip.float().mean()
    huge_top5_drop_rate = huge_top5_drop.float().mean()

    # 7. top10 overlap，防止概率分布尾部大面积乱掉。
    b_top10 = b_top.indices
    c_top10 = c_top.indices
    overlap = (b_top10[..., :, None] == c_top10[..., None, :]).any(dim=-1).float().mean(dim=-1)
    top10_overlap_mean = overlap.mean()

    # 8. centered logits diff，辅助捕捉 logits 形状/scale 的异常。
    b_center = baseline_logits.float() - baseline_logits.float().mean(dim=-1, keepdim=True)
    c_center = candidate_logits.float() - candidate_logits.float().mean(dim=-1, keepdim=True)

    b_center_mean = b_center.mean(dim=1)
    c_center_mean = c_center.mean(dim=1)

    centered_rel_l2 = (
        torch.linalg.vector_norm(c_center_mean - b_center_mean, dim=-1)
        / torch.linalg.vector_norm(b_center_mean, dim=-1).clamp_min(1e-6)
    )
    centered_rel_l2_p99 = torch.quantile(centered_rel_l2.flatten(), 0.99)

    # 9. 硬失败规则。
    checks = {
        "drift_p99_abs": drift_p99 <= abs_js_p99,
        "drift_p999_abs": drift_p999 <= abs_js_p999,
        "drift_vs_baseline_noise": drift_p99 <= alpha * b_noise_p99.clamp_min(1e-12),
        "candidate_noise_vs_baseline": c_noise_p99 <= gamma * b_noise_p99.clamp_min(1e-12),
        "large_margin_flip": large_flip_rate <= max_large_margin_flip_rate,
        "huge_margin_top5_drop": huge_top5_drop_rate <= max_huge_margin_top5_drop_rate,
        "top10_overlap": top10_overlap_mean >= min_top10_overlap,
        "centered_rel_l2": centered_rel_l2_p99 <= max_centered_rel_l2_p99,
    }

    ok = all(bool(v.item() if torch.is_tensor(v) else v) for v in checks.values())

    metrics = {
        "ok": ok,
        "b_noise_p99": float(b_noise_p99),
        "b_noise_p999": float(b_noise_p999),
        "c_noise_p99": float(c_noise_p99),
        "drift_p99": float(drift_p99),
        "drift_p999": float(drift_p999),
        "large_flip_rate": float(large_flip_rate),
        "huge_top5_drop_rate": float(huge_top5_drop_rate),
        "top10_overlap_mean": float(top10_overlap_mean),
        "centered_rel_l2_p99": float(centered_rel_l2_p99),
        "failed_checks": [k for k, v in checks.items() if not bool(v.item() if torch.is_tensor(v) else v)],
    }

    return ok, metrics