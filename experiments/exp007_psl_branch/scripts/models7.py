"""Regional PSL-inspired heads, regularized BES, and metric-Stiefel PUO.

See METHODS.md for explicit differences from the author's pixel-level code.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ("C0", "P1", "P2", "P3", "P4", "P5", "C1")


@torch.no_grad()
def balanced_assignments(similarity, epsilon=0.1, iterations=5):
    """Label-constrained Sinkhorn; rows sum to one, columns approximately balance."""
    n, k = similarity.shape
    if not n:
        return similarity.clone()
    logq = similarity / epsilon
    logq = logq - logq.max()
    q = logq.exp().t().clamp_min(1e-12)
    q /= q.sum()
    for _ in range(iterations):
        q /= q.sum(1, keepdim=True).clamp_min(1e-12)
        q /= k
        q /= q.sum(0, keepdim=True).clamp_min(1e-12)
        q /= n
    return (q * n).t()


@torch.no_grad()
def assign_to_prototypes(z, labels, prototypes, cfg):
    k, d = prototypes.shape[1:]
    assignments = torch.empty(len(z), dtype=torch.long, device=z.device)
    means = prototypes.detach().clone()
    valid = torch.zeros((2, k), dtype=torch.bool, device=z.device)
    p = F.normalize(prototypes.detach(), dim=-1)
    for label in (0, 1):
        subset = labels == label
        if not subset.any():
            continue
        x = z.detach()[subset]
        q = balanced_assignments(x @ p[label].t(), cfg["sinkhorn_epsilon"], cfg["sinkhorn_iterations"])
        assignments[subset] = q.argmax(1) + label * k
        mass = q.sum(0)
        ok = mass > 1e-8
        means[label, ok] = F.normalize((q.t() @ x)[ok] / mass[ok, None], dim=-1)
        valid[label] = ok
    return assignments, means, valid


def scatter_matrices(z, prototypes, assignments):
    """Match source ST/SW construction, retaining the source 2*K denominator."""
    p = prototypes.flatten(0, 1)
    delta = z - p.mean(0)
    total = delta.t() @ delta / max(1, len(z) - 1)
    within = torch.zeros_like(total)
    occupied = 0
    for index, center in enumerate(p):
        x = z[assignments == index]
        if len(x) < 2:
            continue
        delta = x - center
        within = within + delta.t() @ delta / (len(x) - 1)
        occupied += 1
    return total, within / len(p), occupied


def bes_loss(z, prototypes, assignments, cfg):
    """Generalized symmetric eigenspectrum via Cholesky, without unsafe SW^-1."""
    total, within, occupied = scatter_matrices(z, prototypes, assignments)
    if occupied < 2 or len(z) < 4:
        return z.sum() * 0, {"bes_active": 0, "occupied_groups": occupied}
    eye = torch.eye(z.shape[-1], dtype=z.dtype, device=z.device)
    regularized = (within + within.t()) / 2 + cfg["scatter_ridge"] * eye
    lower = torch.linalg.cholesky(regularized)
    between = total - within
    left = torch.linalg.solve_triangular(lower, between, upper=False)
    whitened = torch.linalg.solve_triangular(lower, left.t(), upper=False).t()
    symmetric = (whitened + whitened.t()) / 2
    # Source uses absolute bottom eigenvalues. We retain this choice explicitly;
    # the finite ridge and low-dimensional regional setting are adaptations.
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    bottom = eigenvalues[:min(cfg["bes_bottom_k"], len(eigenvalues))]
    value = 1 / (bottom.abs().sum() + cfg["loss_epsilon"])
    return value, {"bes_active": 1, "occupied_groups": occupied,
                   "scatter_condition": float(torch.linalg.cond(regularized.detach()).item())}


@torch.no_grad()
def polar_columns(matrix):
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh


@torch.no_grad()
def puo_update(prototypes, means, total_scatter, cfg):
    """A Riemannian step and polar retraction in a regularized scatter metric."""
    dimensions = prototypes.shape[-1]
    eye = torch.eye(dimensions, dtype=prototypes.dtype, device=prototypes.device)
    scatter = (total_scatter + total_scatter.t()) / 2 + cfg["scatter_ridge"] * eye
    values, vectors = torch.linalg.eigh(scatter)
    values = values.clamp_min(cfg["scatter_ridge"])
    square_root = (vectors * values.sqrt()[None]) @ vectors.t()
    inverse_root = (vectors * values.rsqrt()[None]) @ vectors.t()
    p = prototypes.flatten(0, 1).t()
    target = means.flatten(0, 1).t()
    q = polar_columns(square_root @ p)
    target = polar_columns(square_root @ target)
    gradient = q - target
    gram = q.t() @ gradient
    tangent = gradient - q @ ((gram + gram.t()) / 2)
    updated = polar_columns(q - cfg["puo_step"] * tangent)
    restored = (inverse_root @ updated).t().reshape_as(prototypes)
    prototypes.copy_(F.normalize(restored, dim=-1))
    return scatter


def supervised_contrastive(z, labels, events, temperature=0.15):
    """Only different-event same-class examples form positive pairs."""
    logits = z @ z.t() / temperature
    identity = torch.eye(len(z), dtype=torch.bool, device=z.device)
    same_class = labels[:, None] == labels[None]
    different_event = events[:, None] != events[None]
    allowed = ~identity & (~same_class | different_event)
    positive = same_class & different_event & ~identity
    valid = positive.any(1)
    if not valid.any():
        return z.sum() * 0
    masked = logits.masked_fill(~allowed, -torch.inf)
    masked = torch.where(allowed.any(1, keepdim=True), masked, torch.zeros_like(masked))
    logprob = logits - torch.logsumexp(masked, dim=1, keepdim=True)
    selected = torch.where(positive, logprob, torch.zeros_like(logprob))
    return -(selected.sum(1)[valid] / positive.sum(1)[valid]).mean()


class RegionalHead(nn.Module):
    def __init__(self, variant, cfg, seed=42):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.cfg = variant, cfg
        d = cfg["projection_dimension"]
        # Fixed streams keep shared projector/fusion initialization identical
        # even when appearance modules have different numbers of parameters.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.projector = nn.Sequential(nn.LayerNorm(cfg["feature_dimension"]),
                                           nn.Linear(cfg["feature_dimension"], d), nn.GELU(),
                                           nn.Linear(d, d))
            torch.manual_seed(seed + 11)
            if variant in ("C0", "C1"):
                self.appearance = nn.Linear(d, 2)
                self.register_parameter("prototypes", None)
            else:
                k = 1 if variant == "P1" else cfg["prototypes_per_class"]
                self.prototypes = nn.Parameter(F.normalize(torch.randn(2, k, d), dim=-1))
            torch.manual_seed(seed + 23)
            self.fusion = nn.Sequential(nn.Linear(d + 4 + 2, cfg["fusion_hidden_dimension"]),
                                        nn.ReLU(), nn.Linear(cfg["fusion_hidden_dimension"], 1))

    def forward(self, features, base):
        z = F.normalize(self.projector(F.normalize(features, dim=-1)), dim=-1)
        if self.prototypes is None:
            appearance = self.appearance(z)
        else:
            p = F.normalize(self.prototypes, dim=-1)
            similarities = torch.einsum("bd,ckd->bck", z, p)
            appearance = similarities.amax(2) / self.cfg["temperature"]
        logit = self.fusion(torch.cat((z, base, appearance), dim=-1)).squeeze(-1)
        return logit, z, appearance

    @torch.no_grad()
    def initialize_prototypes(self, features, labels):
        if self.prototypes is None:
            return
        z = F.normalize(self.projector(F.normalize(features, dim=-1)), dim=-1)
        for label in (0, 1):
            subset = z[labels == label]
            if len(subset) < self.prototypes.shape[1]:
                raise ValueError("Not enough training examples to initialize prototypes")
            # Deterministic class centroid followed by farthest-first examples.
            centers = [F.normalize(subset.mean(0), dim=0)]
            for _ in range(1, self.prototypes.shape[1]):
                similarity = subset @ torch.stack(centers).t()
                centers.append(subset[similarity.amax(1).argmin()])
            self.prototypes[label].copy_(torch.stack(centers))

    @torch.no_grad()
    def normalize_prototypes(self):
        if self.prototypes is not None:
            self.prototypes.copy_(F.normalize(self.prototypes, dim=-1))
