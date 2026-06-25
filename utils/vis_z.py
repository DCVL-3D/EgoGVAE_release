import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_z(save_path, z_head, z_motion):
    N, dim = z_head.shape

    combined_z = torch.cat([z_head, z_motion], dim=0)
    # combined_z = z_head
    combined_z_np = combined_z.cpu().numpy()

    labels = ['head'] * N + ['motion'] * N
    # labels = ['head'] * N

    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(combined_z_np)


    tsne = TSNE(n_components=2, perplexity=30, n_iter_without_progress=300, random_state=42)
    z_tsne = tsne.fit_transform(combined_z_np)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Latent Space Visualization', fontsize=16)

    colors = {'head': 'royalblue', 'motion': 'darkorange'}
    target_names = ['head', 'motion']

    ax1.set_title('PCA Result')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    for name in target_names:
        indices = [i for i, label in enumerate(labels) if label == name]
        ax1.scatter(z_pca[indices, 0], z_pca[indices, 1], c=colors[name], label=name, alpha=0.7)
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('t-SNE Result')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    for name in target_names:
        indices = [i for i, label in enumerate(labels) if label == name]
        ax2.scatter(z_tsne[indices, 0], z_tsne[indices, 1], c=colors[name], label=name, alpha=0.7)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close(fig)
    # plt.show()


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

@torch.no_grad()
def visualize_from_dist_lists(
    head_dist_list,
    motion_dist_list,
    save_path: str,
    total_select: int = 64,       # head+motion 합쳐서 선택할 샘플 수
    samples_per: int = 10,        # 각 샘플당 rsample 횟수
    standardize: bool = True,
    seed: int = 42,
):
    """
    head/motion 분포 리스트(각 원소는 torch.distributions 인스턴스)를 받아
    전체 후보(여러 배치 × 배치내 샘플) 중 총 total_select개를 선택하고,
    각 샘플에서 rsample을 samples_per번 수행하여
    PCA(2D)/t-SNE(2D) 시각화 이미지를 저장합니다.
    - head와 motion은 기본 1:1(절반씩)로 나눠서 선택합니다.
    """
    assert len(head_dist_list) > 0 and len(motion_dist_list) > 0, "분포 리스트가 비어 있습니다."

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 각 분포에서 [S, B, D] 형태로 한 번에 샘플링 후, 배치 차원(B)을 이어붙임
    S = samples_per
    z_head_all = torch.cat([d.rsample((S,)) for d in head_dist_list], dim=1)    # [S, sum_B_head, D]
    z_motion_all = torch.cat([d.rsample((S,)) for d in motion_dist_list], dim=1)# [S, sum_B_motion, D]

    # 선택 수(head/motion) 산정: 기본 1:1 분할
    total_head_max = z_head_all.shape[1]
    total_motion_max = z_motion_all.shape[1]

    head_target = total_select // 2
    motion_target = total_select - head_target

    head_select = min(head_target, total_head_max)
    motion_select = min(motion_target, total_motion_max)

    # 혹시 한쪽이 부족하면 남은 몫을 다른 쪽에서 보충
    remaining = total_select - (head_select + motion_select)
    if remaining > 0:
        # head에 여유가 있으면 head에서 보충
        extra_head = min(remaining, total_head_max - head_select)
        head_select += extra_head
        remaining -= extra_head
    if remaining > 0:
        # 그래도 남으면 motion에서 보충
        extra_motion = min(remaining, total_motion_max - motion_select)
        motion_select += extra_motion
        remaining -= extra_motion

    # 실제 선택 슬라이스 (배치축에서 앞쪽부터)
    z_head = z_head_all[:, :head_select, ...]       # [S, head_select, D]
    z_motion = z_motion_all[:, :motion_select, ...] # [S, motion_select, D]

    # (S, K, D) -> (S*K, D)
    z_head = z_head.reshape(-1, z_head.shape[-1]).detach().float().cpu().numpy()
    z_motion = z_motion.reshape(-1, z_motion.shape[-1]).detach().float().cpu().numpy()

    # 결합 및 라벨
    combined = np.concatenate([z_head, z_motion], axis=0)
    labels = (["head"] * z_head.shape[0]) + (["motion"] * z_motion.shape[0])

    # (권장) 표준화
    if standardize:
        scaler = StandardScaler()
        combined_std = scaler.fit_transform(combined)
    else:
        combined_std = combined

    # PCA(2D)
    pca_2 = PCA(n_components=2, random_state=seed)
    z_pca_2 = pca_2.fit_transform(combined_std)

    # t-SNE 가속을 위한 PCA 50D 축소
    feat_dim = combined_std.shape[1]
    pca_50 = PCA(n_components=min(50, feat_dim), random_state=seed)
    z_pca_50 = pca_50.fit_transform(combined_std)

    # t-SNE perplexity 자동 조정 (샘플 수에 따른 안전 범위)
    n_points = combined_std.shape[0]
    perplexity = min(30, max(5, n_points // 3 - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        n_iter_without_progress=300,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="euclidean",
    )
    z_tsne_2 = tsne.fit_transform(z_pca_50)

    # 플롯
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    title = (f"Latent Visualization  |  select={head_select}+{motion_select}={head_select+motion_select}, "
             f"S={S} → points={n_points}")
    fig.suptitle(title, fontsize=15)

    color_map = {"head": "royalblue", "motion": "darkorange"}

    def scatter(ax, emb, ttl):
        ax.set_title(ttl)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        idx_head = [i for i, lb in enumerate(labels) if lb == "head"]
        idx_motion = [i for i, lb in enumerate(labels) if lb == "motion"]
        ax.scatter(emb[idx_head, 0], emb[idx_head, 1], c=color_map["head"], label="head", alpha=0.7, s=18)
        ax.scatter(emb[idx_motion, 0], emb[idx_motion, 1], c=color_map["motion"], label="motion", alpha=0.7, s=18)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

    scatter(ax1, z_pca_2, "PCA (2D)")
    scatter(ax2, z_tsne_2, f"t-SNE (2D)  (perplexity={perplexity})")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)



@torch.no_grad()
def visualize_from_dist_lists_with_prior(
    head_dist_list,
    motion_dist_list,
    save_path: str,
    total_select: int = 64,       # head+motion 합쳐서 선택할 샘플 수 (기본 64 → head 32, motion 32)
    samples_per: int = 10,        # 각 샘플 당 rsample 횟수
    include_prior: bool = True,   # 표준정규 prior 시각화 포함 여부
    prior_match_total: bool = True,  # prior 포인트 수를 head+motion 합과 동일하게 맞출지
    standardize: bool = True,
    seed: int = 42,
):
    """
    여러 배치에 걸친 head/motion 분포 리스트에서 총 total_select개만 선택하고,
    각 샘플에 대해 rsample()을 samples_per번 수행하여 PCA/t-SNE로 시각화.
    include_prior=True이면 표준정규분포 N(0,I)에서 동일 스케일의 샘플도 함께 시각화.
    - 기본: head 32 + motion 32 선택, 각 10회 rsample → head 320, motion 320, prior 640 (합 1280)
    - prior_match_total=True: prior 포인트 수 = head+motion 포인트 수 로 맞춤
    """
    assert len(head_dist_list) > 0 and len(motion_dist_list) > 0, "분포 리스트가 비어 있습니다."

    torch.manual_seed(seed)
    np.random.seed(seed)

    S = samples_per

    # 1) 리스트의 모든 배치를 한 번에 샘플링: [S, sum_B, D]
    z_head_all = torch.cat([d.rsample((S,)) for d in head_dist_list], dim=1)     # [S, B_h, D]
    z_motion_all = torch.cat([d.rsample((S,)) for d in motion_dist_list], dim=1) # [S, B_m, D]

    # 2) head/motion에서 선택할 개수(절반씩, 부족 시 보충)
    total_head_max = z_head_all.shape[1]
    total_motion_max = z_motion_all.shape[1]
    latent_dim = z_head_all.shape[-1]

    head_target = total_select // 2
    motion_target = total_select - head_target

    head_select = min(head_target, total_head_max)
    motion_select = min(motion_target, total_motion_max)

    remaining = total_select - (head_select + motion_select)
    if remaining > 0:
        extra_head = min(remaining, total_head_max - head_select)
        head_select += extra_head
        remaining -= extra_head
    if remaining > 0:
        extra_motion = min(remaining, total_motion_max - motion_select)
        motion_select += extra_motion
        remaining -= extra_motion

    # 3) 앞에서부터 선택해 펼치기: (S, K, D) → (S*K, D)
    z_head = z_head_all[:, :head_select].reshape(-1, latent_dim).detach().float().cpu().numpy()
    z_motion = z_motion_all[:, :motion_select].reshape(-1, latent_dim).detach().float().cpu().numpy()

    # 4) prior 샘플 생성: 표준정규(N(0,I))
    prior_points = None
    if include_prior:
        # 기본적으로 prior 포인트 수를 head+motion 합과 동일하게 맞춤
        prior_count = z_head.shape[0]
        # torch.randn으로 직접 생성
        prior_points = torch.randn(prior_count, latent_dim).cpu().numpy()

    # 5) 결합 및 라벨
    parts = [z_head, z_motion]
    labels = (["head"] * z_head.shape[0]) + (["motion"] * z_motion.shape[0])
    if include_prior and prior_points is not None and prior_points.size > 0:
        parts.append(prior_points)
        labels += (["prior"] * prior_points.shape[0])

    combined = np.concatenate(parts, axis=0)

    # 6) (권장) 표준화 후 차원축소
    X = StandardScaler().fit_transform(combined) if standardize else combined

    pca_2 = PCA(n_components=2, random_state=seed)
    X_pca2 = pca_2.fit_transform(X)

    # t-SNE 가속/안정화를 위한 50D 사전 축소
    feat_dim = X.shape[1]
    pca_50 = PCA(n_components=min(50, feat_dim), random_state=seed)
    X_pca50 = pca_50.fit_transform(X)

    # 7) t-SNE perplexity 안전 범위로 자동 조정
    n_points = X.shape[0]
    perplexity = min(30, max(5, n_points // 3 - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        n_iter_without_progress=300,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="euclidean",
    )
    X_tsne2 = tsne.fit_transform(X_pca50)

    # 8) 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    title = (
        f"Latent Visualization | select head={head_select}, motion={motion_select}, S={S} "
        f"→ points H={z_head.shape[0]}, M={z_motion.shape[0]}"
        + (f", P={prior_points.shape[0]}" if include_prior and prior_points is not None else "")
    )
    fig.suptitle(title, fontsize=14)

    color_map = {"head": "royalblue", "motion": "darkorange", "prior": "seagreen"}

    def scatter(ax, emb, ttl):
        ax.set_title(ttl)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        for name in ["head", "motion", "prior"]:
            idx = [i for i, lb in enumerate(labels) if lb == name]
            if len(idx) == 0:
                continue
            ax.scatter(emb[idx, 0], emb[idx, 1], c=color_map[name], label=name, alpha=0.7, s=16)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

    scatter(ax1, X_pca2, "PCA (2D)")
    scatter(ax2, X_tsne2, f"t-SNE (2D) (perplexity={perplexity})")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=220)
    plt.close(fig)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

@torch.no_grad()
def visualize_tsne_rsample_dense(
    head_dist_list,
    motion_dist_list,
    save_path: str,
    samples_per_dist: int = 5,         # 각 dist에서 몇 번 rsample할지
    max_points_per_class: int = 20000, # head/motion/prior 각각 최대 포인트 수
    standardize: bool = True,
    seed: int = 42,
    return_embedding: bool = False,
):
    """
    head_dist_list, motion_dist_list:
        각 원소가 torch.distributions.Normal (또는 rsample() 있는 분포)인 리스트.

    - head/motion 분포에서 rsample을 여러 번 뽑고 (samples_per_dist),
    - 같은 latent 차원과 샘플 수를 가지는 표준 정규분포 N(0, I)를 prior로 생성해서
      head/motion/prior를 한 번에 t-SNE로 시각화.
    - PCA 50D 같은 거 안 쓰고, (선택적으로) StandardScaler -> t-SNE만 수행.
    """
    assert len(head_dist_list) > 0 and len(motion_dist_list) > 0, "head/motion 분포 리스트가 비어 있습니다."

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 고정 컬러 팔레트
    color_map = {
        "head": "royalblue",
        "motion": "darkorange",
        "prior": "seagreen",
    }

    # ---------- 공통: rsample + reshape ----------
    def collect_z_from_dists(dist_list):
        z_list = []
        for d in dist_list:
            z = d.rsample((samples_per_dist,))  # (S, B, D) 또는 (S, B, T, D) 등
            if z.ndim == 4:
                S, B, T, D = z.shape
                z = z.reshape(S * B * T, D)
            elif z.ndim == 3:
                S, B, D = z.shape
                z = z.reshape(S * B, D)
            else:
                z = z.reshape(-1, z.shape[-1])
            z_list.append(z.detach().cpu())
        return torch.cat(z_list, dim=0)  # (N_total, D)

    def subsample(x: torch.Tensor, max_points: int) -> torch.Tensor:
        N = x.shape[0]
        if N <= max_points:
            return x
        idx = torch.randperm(N)[:max_points]
        return x[idx]

    # ---------- 1) head / motion rsample ----------
    head_z_all   = collect_z_from_dists(head_dist_list)      # (N_head, D)
    motion_z_all = collect_z_from_dists(motion_dist_list)    # (N_motion, D)

    # prior는 "head와 같은 크기/latent 차원"의 표준 정규분포로 생성
    prior_z_all  = torch.randn_like(head_z_all)              # (N_head, D)

    # 각 클래스별 포인트 수 제한
    head_z   = subsample(head_z_all,   max_points_per_class)
    motion_z = subsample(motion_z_all, max_points_per_class)
    prior_z  = subsample(prior_z_all,  max_points_per_class)

    head_np   = head_z.numpy()
    motion_np = motion_z.numpy()
    prior_np  = prior_z.numpy()

    # ---------- 2) 결합 + 라벨 ----------
    combined = np.concatenate([head_np, motion_np, prior_np], axis=0)
    n_head   = head_np.shape[0]
    n_motion = motion_np.shape[0]
    n_prior  = prior_np.shape[0]

    labels = (
        [0] * n_head +
        [1] * n_motion +
        [2] * n_prior
    )
    labels = np.array(labels)

    # ---------- 3) (선택) 표준화 ----------
    if standardize:
        # prior가 스케일을 망가뜨리지 않게 head+motion만으로 scaler를 fit
        scaler = StandardScaler()
        hm = np.concatenate([head_np, motion_np], axis=0)
        scaler.fit(hm)
        head_std   = scaler.transform(head_np)
        motion_std = scaler.transform(motion_np)
        prior_std  = scaler.transform(prior_np)
        combined_std = np.concatenate([head_std, motion_std, prior_std], axis=0)
    else:
        combined_std = combined

    # ---------- 4) t-SNE ----------
    n_points = combined_std.shape[0]
    perplexity = min(30, max(5, n_points // 3 - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        max_iter=1000,
        init="random",
        metric="euclidean",
        random_state=seed,
    )
    z_2d = tsne.fit_transform(combined_std)  # (N_total, 2)

    head_2d   = z_2d[labels == 0]
    motion_2d = z_2d[labels == 1]
    prior_2d  = z_2d[labels == 2]

    # ---------- 5) 시각화 ----------
    plt.figure(figsize=(7, 7))
    plt.scatter(
        head_2d[:, 0], head_2d[:, 1],
        s=16, alpha=0.7,
        c=color_map["head"],
        label="head",
    )
    plt.scatter(
        motion_2d[:, 0], motion_2d[:, 1],
        s=16, alpha=0.7,
        c=color_map["motion"],
        label="motion",
    )
    plt.scatter(
        prior_2d[:, 0], prior_2d[:, 1],
        s=16, alpha=0.7,
        c=color_map["prior"],
        label="prior",
    )

    title = (
        f"t-SNE (rsample dense) | "
        f"head={n_head}, motion={n_motion}, prior={n_prior}, "
        f"S={samples_per_dist}, points={n_points}"
    )
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[visualize_tsne_rsample_dense] Saved to {save_path}")

    if return_embedding:
        return z_2d, labels

def align_to_reference(Z_cur: np.ndarray, Z_ref: np.ndarray):
    """
    Z_cur: (N_cur, 2) 현재 epoch의 prior 2D
    Z_ref: (N_ref, 2) 기준(epoch10)의 prior 2D
    """
    # 간단히 작은 쪽 개수에 맞춰서 사용
    N = min(len(Z_cur), len(Z_ref))
    Z_cur = Z_cur[:N]
    Z_ref = Z_ref[:N]

    mu_cur = Z_cur.mean(axis=0, keepdims=True)
    mu_ref = Z_ref.mean(axis=0, keepdims=True)
    X = Z_cur - mu_cur
    Y = Z_ref - mu_ref

    U, _, VT = np.linalg.svd(X.T @ Y)
    R = U @ VT              # (2, 2)
    t = mu_ref - mu_cur @ R # (1, 2)

    return R, t


@torch.no_grad()
def visualize_tsne_rsample_dense_aligned_to_ref(
    head_dist_list,
    motion_dist_list,
    prior_ref_path: str,          # ✅ epoch10에서 저장한 prior_2d npy
    save_path_aligned: str,       # 정렬 후 그림 저장 경로
    samples_per_dist: int = 5,
    max_points_per_class: int = 20000,
    standardize: bool = True,
    seed: int = 42,
):
    """
    - 먼저 visualize_tsne_rsample_dense를 사용해 현재 epoch의 t-SNE 임베딩(z_2d, labels)을 얻고,
    - prior_ref_path에 저장된 기준 prior(epoch10)을 불러와
      prior 기준으로 회전+이동 정렬한 뒤,
      정렬된 임베딩으로 다시 그림을 저장.
    """
    # 1) 기준 prior (epoch10) 로드
    prior_ref_2d = np.load(prior_ref_path)  # (N_ref, 2)

    # 2) 현재 epoch의 t-SNE 임베딩 얻기 (raw 이미지도 원하면 저장)
    z_2d, labels = visualize_tsne_rsample_dense(
        head_dist_list,
        motion_dist_list,
        save_path=save_path_aligned.replace(".png", "_raw.png"),
        samples_per_dist=samples_per_dist,
        max_points_per_class=max_points_per_class,
        standardize=standardize,
        seed=seed,
        return_embedding=True,
    )

    # 3) 현재 epoch prior 좌표 추출
    prior_cur_2d = z_2d[labels == 2]

    # 4) epoch10 기준으로 정렬 변환 R, t 계산
    R, t = align_to_reference(prior_cur_2d, prior_ref_2d)

    # 5) 전체 포인트(head/motion/prior)를 정렬된 좌표계로 변환
    z_2d_aligned = z_2d @ R + t   # (N_total, 2)

    # 6) 정렬된 좌표로 다시 색깔 입혀서 시각화
    color_map = {
        "head": "royalblue",
        "motion": "darkorange",
        "prior": "seagreen",
    }

    head_2d   = z_2d_aligned[labels == 0]
    motion_2d = z_2d_aligned[labels == 1]
    prior_2d  = z_2d_aligned[labels == 2]

    plt.figure(figsize=(7, 7))
    plt.scatter(
        head_2d[:, 0], head_2d[:, 1],
        s=16, alpha=0.7,
        c=color_map["head"],
        label=r"$\mathcal{N}(\mu^H,\Sigma^H)$",
    )
    plt.scatter(
        motion_2d[:, 0], motion_2d[:, 1],
        s=16, alpha=0.7,
        c=color_map["motion"],
        label=r"$\mathcal{N}(\mu^M,\Sigma^M)$",
    )
    plt.scatter(
        prior_2d[:, 0], prior_2d[:, 1],
        s=16, alpha=0.7,
        c=color_map["prior"],
        label=r"$\mathcal{N}(0,\mathrm{I})$",
    )

    # plt.legend(
    #     # loc="upper right",
    #     # bbox_to_anchor=(1.0, 1.0),
    #     fontsize=15,
    #     markerscale=2.0,
    # )
    # plt.title("Aligned t-SNE (aligned to reference prior)")
    # plt.xlabel("dim 1")
    # plt.ylabel("dim 2")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path_aligned) or ".", exist_ok=True)
    plt.savefig(save_path_aligned, dpi=300)
    plt.close()
    print(f"[visualize_tsne_rsample_dense_aligned_to_ref] Saved to {save_path_aligned}")

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from pathlib import Path

def save_axis_npz(save_path, xlim, ylim):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, xlim=np.array(xlim), ylim=np.array(ylim))

def load_axis_npz(load_path):
    d = np.load(load_path)
    xlim = tuple(d["xlim"].tolist())
    ylim = tuple(d["ylim"].tolist())
    return xlim, ylim

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne_rsample_with_axis_file(
    head_dist_list,
    motion_dist_list,
    out_png_path,
    axis_npz_path,
    *,
    mode="write",          # "write" or "read"
    K=8,
    perplexity=30,
    random_state=0,
    percentile=1.0,
    pad_ratio=0.05,
    point_size=5,
    alpha=0.6,
    dpi=200,
):
    # -------------------------
    # 1) rsample -> X 만들기
    # -------------------------
    hs, ms = [], []
    for hd, md in zip(head_dist_list, motion_dist_list):
        hs.append(hd.rsample((K,)).reshape(-1, hd.mean.shape[-1]).detach().cpu())
        ms.append(md.rsample((K,)).reshape(-1, md.mean.shape[-1]).detach().cpu())

    hs = torch.cat(hs, dim=0)
    ms = torch.cat(ms, dim=0)

    # -------------------------
    # 1-1) Normal distribution 내부 생성
    # -------------------------
    latent_dim = hs.shape[-1]
    num_samples = hs.shape[0]  # head와 개수 맞춤

    normal = torch.randn(num_samples, latent_dim)

    # numpy 변환
    hs = hs.numpy()
    ms = ms.numpy()
    normal = normal.numpy()

    X = np.concatenate([hs, ms, normal], axis=0)
    labels = np.concatenate([
        np.zeros(len(hs), dtype=np.int32),      # head
        np.ones(len(ms), dtype=np.int32),       # motion
        np.full(len(normal), 2, dtype=np.int32) # normal
    ])

    # -------------------------
    # 2) t-SNE
    # -------------------------
    Y = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(X)

    Y_head = Y[labels == 0]
    Y_motion = Y[labels == 1]
    Y_normal = Y[labels == 2]

    # -------------------------
    # 3) axis 결정
    # -------------------------
    if mode == "write":
        xmin, ymin = np.percentile(Y, percentile, axis=0)
        xmax, ymax = np.percentile(Y, 100 - percentile, axis=0)

        dx = max(xmax - xmin, 1e-8)
        dy = max(ymax - ymin, 1e-8)

        padx = pad_ratio * dx
        pady = pad_ratio * dy

        xlim = (xmin - padx, xmax + padx)
        ylim = (ymin - pady, ymax + pady)

        np.savez(axis_npz_path, xlim=xlim, ylim=ylim)

    elif mode == "read":
        data = np.load(axis_npz_path)
        xlim = tuple(data["xlim"])
        ylim = tuple(data["ylim"])
    else:
        raise ValueError('mode must be "write" or "read"')

    # -------------------------
    # 4) plot + save
    # -------------------------
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(Y_head[:, 0], Y_head[:, 1], s=point_size, alpha=alpha)
    plt.scatter(Y_motion[:, 0], Y_motion[:, 1], s=point_size, alpha=alpha)
    plt.scatter(Y_normal[:, 0], Y_normal[:, 1], s=point_size, alpha=alpha)

    plt.axis("equal")
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=dpi)
    plt.close()

    return xlim, ylim


import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def align_to_reference(Z_cur: np.ndarray, Z_ref: np.ndarray):
    """
    Z_cur: (N_cur, 2) 현재 epoch의 prior 2D
    Z_ref: (N_ref, 2) 기준(epoch10 등)의 prior 2D
    """
    N = min(len(Z_cur), len(Z_ref))
    Z_cur = Z_cur[:N]
    Z_ref = Z_ref[:N]

    mu_cur = Z_cur.mean(axis=0, keepdims=True)
    mu_ref = Z_ref.mean(axis=0, keepdims=True)
    X = Z_cur - mu_cur
    Y = Z_ref - mu_ref

    U, _, VT = np.linalg.svd(X.T @ Y)
    R = U @ VT              # (2, 2)
    t = mu_ref - mu_cur @ R # (1, 2)

    return R, t


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne_rsample_with_axis_file_align(
    head_dist_list,
    motion_dist_list,
    out_png_path,
    axis_npz_path=None, # 이제 필수가 아닐 수 있으므로 기본값 None
    *,
    mode="write",
    K=8,
    perplexity=30,
    random_state=0,
    percentile=1.0,
    pad_ratio=0.05,
    point_size=5,
    alpha=0.6,
    dpi=200,
    prior_ref_path: str | None = None,
    fixed_xlim: tuple | None = (-70, 70), # ✅ 추가: 기본값 -60~60
    fixed_ylim: tuple | None = (-70, 70), # ✅ 추가: 기본값 -60~60
):
    # -------------------------
    # 고정 컬러 팔레트
    # -------------------------
    color_map = {
        "head": "royalblue",
        "motion": "darkorange",
        "prior": "seagreen",
    }

    # 1) rsample (기존 로직 동일)
    hs, ms = [], []
    for hd, md in zip(head_dist_list, motion_dist_list):
        hs.append(hd.rsample((K,)).reshape(-1, hd.mean.shape[-1]).detach().cpu())
        ms.append(md.rsample((K,)).reshape(-1, md.mean.shape[-1]).detach().cpu())

    hs = torch.cat(hs, dim=0).numpy()
    ms = torch.cat(ms, dim=0).numpy()
    num_samples = hs.shape[0]
    prior = torch.randn(num_samples, hs.shape[-1]).numpy()

    X = np.concatenate([hs, ms, prior], axis=0)
    labels = np.concatenate([
        np.zeros(len(hs), dtype=np.int32),
        np.ones(len(ms), dtype=np.int32),
        np.full(len(prior), 2, dtype=np.int32),
    ])

    # 2) t-SNE
    Y = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(X)

    # 2-1) align to reference (기존 로직 동일)
    if prior_ref_path is not None:
        prior_ref_2d = np.load(prior_ref_path)
        prior_cur_2d = Y[labels == 2]
        R, t = align_to_reference(prior_cur_2d, prior_ref_2d)
        Y = Y @ R + t

    Y_head = Y[labels == 0]
    Y_motion = Y[labels == 1]
    Y_prior = Y[labels == 2]

    # -------------------------
    # 3) axis 설정 (수정된 부분)
    # -------------------------
    if fixed_xlim is not None and fixed_ylim is not None:
        # 사용자가 직접 범위를 지정한 경우 (우선순위 1순위)
        xlim, ylim = fixed_xlim, fixed_ylim
    elif mode == "write":
        xmin, ymin = np.percentile(Y, percentile, axis=0)
        xmax, ymax = np.percentile(Y, 100 - percentile, axis=0)
        dx, dy = max(xmax - xmin, 1e-8), max(ymax - ymin, 1e-8)
        padx, pady = pad_ratio * dx, pad_ratio * dy
        xlim = (xmin - padx, xmax + padx)
        ylim = (ymin - pady, ymax + pady)
        if axis_npz_path:
            np.savez(axis_npz_path, xlim=xlim, ylim=ylim)
    elif mode == "read":
        data = np.load(axis_npz_path)
        xlim, ylim = tuple(data["xlim"]), tuple(data["ylim"])
    else:
        raise ValueError('Invalid mode or missing limits')

    # ... (생략: t-SNE 계산 및 데이터 분리 로직)

    # -------------------------
    # 4) plot
    # -------------------------
    os.makedirs(os.path.dirname(out_png_path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 6))
    
    # 축 숫자 크기
    tick_label_size = 35
    plt.tick_params(axis='both', which='major', labelsize=tick_label_size)

    # ✅ [수정 포인트] 원하는 눈금을 리스트로 직접 지정
    # -70, 70 범위 내에서 숫자는 딱 이 값들만 표시됩니다.
    fixed_ticks = [-60, -30, 0, 30, 60]
    plt.xticks(fixed_ticks)
    plt.yticks(fixed_ticks)

    # 데이터 플롯
    plt.scatter(Y_head[:, 0], Y_head[:, 1], s=point_size, alpha=alpha, c=color_map["head"])
    plt.scatter(Y_motion[:, 0], Y_motion[:, 1], s=point_size, alpha=alpha, c=color_map["motion"])
    plt.scatter(Y_prior[:, 0], Y_prior[:, 1], s=point_size, alpha=alpha, c=color_map["prior"])

    # ✅ 축 범위 설정 (-70, 70)
    plt.axis("equal") 
    plt.xlim(-70, 70)
    plt.ylim(-70, 70)

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=dpi)
    plt.close()

def plot_tsne_rsample_with_axis_file_align_wo_p(
    head_dist_list,
    motion_dist_list,
    out_png_path,
    axis_npz_path=None, # 이제 필수가 아닐 수 있으므로 기본값 None
    *,
    mode="write",
    K=8,
    perplexity=30,
    random_state=0,
    percentile=1.0,
    pad_ratio=0.05,
    point_size=5,
    alpha=0.6,
    dpi=200,
    use_mean = False,
    prior_ref_path: str | None = None,
    fixed_xlim: tuple | None = (-60, 60), # ✅ 추가: 기본값 -60~60
    fixed_ylim: tuple | None = (-60, 60), # ✅ 추가: 기본값 -60~60
    
):
    # -------------------------
    # 고정 컬러 팔레트
    # -------------------------
    color_map = {
        "head": "royalblue",
        "motion": "darkorange",
        "prior": "seagreen",
    }

    # 1) rsample (기존 로직 동일)
    hs, ms = [], []
    for hd, md in zip(head_dist_list, motion_dist_list):
        if use_mean:
            h = hd.mean.reshape(-1, hd.mean.shape[-1])
            m = md.mean.reshape(-1, md.mean.shape[-1])
        else:
            h = hd.rsample((K,)).reshape(-1, hd.mean.shape[-1])
            m = md.rsample((K,)).reshape(-1, md.mean.shape[-1])

        hs.append(h.detach().cpu())
        ms.append(m.detach().cpu())

    hs = torch.cat(hs, dim=0).numpy()
    ms = torch.cat(ms, dim=0).numpy()
    # num_samples = hs.shape[0]
    # prior = torch.randn(num_samples, hs.shape[-1]).numpy()

    X = np.concatenate([hs, ms], axis=0)
    labels = np.concatenate([
        np.zeros(len(hs), dtype=np.int32),
        np.ones(len(ms), dtype=np.int32),
    ])

    # 2) t-SNE
    Y = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(X)

    # 2-1) align to reference (기존 로직 동일)
    if prior_ref_path is not None:
        prior_ref_2d = np.load(prior_ref_path)
        prior_cur_2d = Y[labels == 0]
        R, t = align_to_reference(prior_cur_2d, prior_ref_2d)
        Y = Y @ R + t

    Y_head = Y[labels == 0]
    Y_motion = Y[labels == 1]
    # Y_prior = Y[labels == 2]

    # -------------------------
    # 3) axis 설정 (수정된 부분)
    # -------------------------
    if fixed_xlim is not None and fixed_ylim is not None:
        # 사용자가 직접 범위를 지정한 경우 (우선순위 1순위)
        xlim, ylim = fixed_xlim, fixed_ylim
    elif mode == "write":
        xmin, ymin = np.percentile(Y, percentile, axis=0)
        xmax, ymax = np.percentile(Y, 100 - percentile, axis=0)
        dx, dy = max(xmax - xmin, 1e-8), max(ymax - ymin, 1e-8)
        padx, pady = pad_ratio * dx, pad_ratio * dy
        xlim = (xmin - padx, xmax + padx)
        ylim = (ymin - pady, ymax + pady)
        if axis_npz_path:
            np.savez(axis_npz_path, xlim=xlim, ylim=ylim)
    elif mode == "read":
        data = np.load(axis_npz_path)
        xlim, ylim = tuple(data["xlim"]), tuple(data["ylim"])
    else:
        raise ValueError('Invalid mode or missing limits')

    # ... (생략: t-SNE 계산 및 데이터 분리 로직)

    # -------------------------
    # 4) plot
    # -------------------------
    os.makedirs(os.path.dirname(out_png_path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 6))
    
    # 축 숫자 크기
    tick_label_size = 35
    plt.tick_params(axis='both', which='major', labelsize=tick_label_size)

    # ✅ [수정 포인트] 원하는 눈금을 리스트로 직접 지정
    # -70, 70 범위 내에서 숫자는 딱 이 값들만 표시됩니다.
    fixed_ticks = [ -50, -30, 0, 30, 50]
    plt.xticks(fixed_ticks)
    plt.yticks(fixed_ticks)

    # 데이터 플롯
    plt.scatter(Y_head[:, 0], Y_head[:, 1], s=point_size, alpha=alpha, c=color_map["head"])
    plt.scatter(Y_motion[:, 0], Y_motion[:, 1], s=point_size, alpha=alpha, c=color_map["motion"])
    # plt.scatter(Y_prior[:, 0], Y_prior[:, 1], s=point_size, alpha=alpha, c=color_map["prior"])

    # ✅ 축 범위 설정 (-70, 70)
    plt.axis("equal") 
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=dpi)
    plt.close()