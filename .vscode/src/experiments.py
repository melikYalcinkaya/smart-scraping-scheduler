"""
Ana Deney Koşucusu
===================
Tüm deneyleri çalıştırır ve sonuçları kaydeder:

Deney 1: Temel karşılaştırma (5 algoritma vs baseline)
Deney 2: Parametre duyarlılık analizi (her algoritma için)
Deney 3: Farklı bütçe kısıtlamaları altında performans
Deney 4: Farklı site sayıları ile ölçeklenebilirlik
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from environment import ECommerceEnvironment
from algorithms import (
    PeriodicScheduler, AdaptiveCrawler,
    PriorityQueueScheduler, PoissonMLEScheduler, UCBBanditScheduler
)
from metrics import Evaluator, ExperimentResult

# ── Genel ayarlar ──────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "Periodic (Baseline)":  "#95a5a6",
    "Adaptive Crawling":    "#3498db",
    "Priority Queue":       "#e74c3c",
    "Poisson MLE":          "#2ecc71",
    "UCB Bandit (c=1.0)":   "#9b59b6",
    "UCB Bandit (c=0.5)":   "#e67e22",
    "UCB Bandit (c=2.0)":   "#1abc9c",
}

def get_color(name: str) -> str:
    for key, color in COLORS.items():
        if key in name:
            return color
    return "#34495e"


# ══════════════════════════════════════════════════════════════
# DENEY 1: Temel Karşılaştırma
# ══════════════════════════════════════════════════════════════

def experiment_1_baseline_comparison():
    print("\n" + "="*60)
    print("DENEY 1: Temel Algoritma Karşılaştırması")
    print("="*60)

    env = ECommerceEnvironment(n_sites=20, seed=42)
    evaluator = Evaluator(env, n_steps=500, time_step=0.5)

    schedulers = [
        PeriodicScheduler(n_sites=20, budget_per_step=3),
        AdaptiveCrawler(n_sites=20, budget_per_step=3),
        PriorityQueueScheduler(n_sites=20, budget_per_step=3),
        PoissonMLEScheduler(n_sites=20, budget_per_step=3),
        UCBBanditScheduler(n_sites=20, budget_per_step=3, exploration_c=1.0),
    ]

    results = []
    for sched in schedulers:
        print(f"  Çalışıyor: {sched.name}...", end=" ", flush=True)
        res = evaluator.run(sched)
        results.append(res)
        print(f"✓ ({res.wall_clock_time:.2f}s)")

    # Tablo
    df = pd.DataFrame([r.summary() for r in results])
    df.to_csv(RESULTS_DIR / "exp1_comparison.csv", index=False)
    print("\n📊 Sonuçlar:")
    print(df[["Algorithm", "Data Freshness Rate", "Crawl Efficiency",
              "Change Detection Rate", "Wasted Crawl Ratio", "Avg Staleness"]].to_string(index=False))

    # Grafik 1: Çok metrikli karşılaştırma
    _plot_exp1(results, df)
    return results, df


def _plot_exp1(results, df):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    metrics = [
        ("Data Freshness Rate",   "Veri Güncellik Oranı\n(↑ İyi)",   "Blues_d"),
        ("Crawl Efficiency",      "Tarama Verimliliği\n(Maliyet/Değişim, ↓ İyi)", "Reds_d"),
        ("Change Detection Rate", "Değişim Yakalama Oranı\n(↑ İyi)", "Greens_d"),
        ("Wasted Crawl Ratio",    "Boş Tarama Oranı\n(↓ İyi)",       "Oranges_d"),
        ("Avg Staleness",         "Ortalama Güncellik Kaybı\n(↓ İyi)", "Purples_d"),
        ("Total Resource Cost",   "Toplam Kaynak Maliyeti\n(↓ İyi)",  "Greys_d"),
    ]

    algo_names = [r.algorithm_name for r in results]
    short_names = [n.replace(" (Baseline)", "\n(Baseline)")
                    .replace(" Crawling", "\nCrawling")
                    .replace(" Queue", "\nQueue")
                    .replace(" MLE", "\nMLE")
                    .replace("UCB ", "UCB\n") for n in algo_names]

    for idx, (metric, title, cmap) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        vals = df[metric].values
        colors = [get_color(n) for n in algo_names]
        bars = ax.bar(range(len(algo_names)), vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(algo_names)))
        ax.set_xticklabels(short_names, fontsize=7.5)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(metric.split("(")[0].strip(), fontsize=8)
        # Değerleri çubukların üstüne yaz
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle("Deney 1: 5 Algoritma Temel Karşılaştırması\n(20 site, 500 adım, bütçe=3)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(RESULTS_DIR / "exp1_comparison.png", bbox_inches="tight")
    plt.close()
    print("  → exp1_comparison.png kaydedildi")

    # Staleness zaman serisi
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for res in results:
        steps = np.arange(len(res.staleness_history))
        ax2.plot(steps, res.staleness_history, label=res.algorithm_name,
                 color=get_color(res.algorithm_name), linewidth=1.8, alpha=0.85)
    ax2.set_xlabel("Adım")
    ax2.set_ylabel("Toplam Güncellik Kaybı (Staleness)")
    ax2.set_title("Zaman İçinde Güncellik Kaybı Karşılaştırması", fontweight="bold")
    ax2.legend(framealpha=0.9)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp1_staleness_timeline.png", bbox_inches="tight")
    plt.close()
    print("  → exp1_staleness_timeline.png kaydedildi")


# ══════════════════════════════════════════════════════════════
# DENEY 2: Parametre Duyarlılık Analizi
# ══════════════════════════════════════════════════════════════

def experiment_2_parameter_sensitivity():
    print("\n" + "="*60)
    print("DENEY 2: Parametre Duyarlılık Analizi")
    print("="*60)

    env = ECommerceEnvironment(n_sites=20, seed=42)
    evaluator = Evaluator(env, n_steps=500, time_step=0.5)
    all_rows = []

    # 2a. Adaptive: base_interval duyarlılığı
    print("  2a. Adaptive Crawling - base_interval...")
    intervals = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    adapt_results = []
    for bi in intervals:
        sched = AdaptiveCrawler(n_sites=20, budget_per_step=3, base_interval=bi)
        res = evaluator.run(sched)
        adapt_results.append((bi, res))
        row = res.summary()
        row["param_name"] = "base_interval"
        row["param_value"] = bi
        row["algorithm_group"] = "Adaptive"
        all_rows.append(row)

    # 2b. Adaptive: history_window duyarlılığı
    print("  2b. Adaptive Crawling - history_window...")
    windows = [3, 5, 10, 20, 50]
    for w in windows:
        sched = AdaptiveCrawler(n_sites=20, budget_per_step=3, history_window=w)
        res = evaluator.run(sched)
        row = res.summary()
        row["param_name"] = "history_window"
        row["param_value"] = w
        row["algorithm_group"] = "Adaptive"
        all_rows.append(row)

    # 2c. Priority Queue: alpha/beta/gamma
    print("  2c. Priority Queue - alpha/beta ağırlıkları...")
    pq_params = [
        (1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 1.0),
        (1.0, 1.0, 2.0), (0.5, 2.0, 0.5), (2.0, 2.0, 0.5)
    ]
    for a, b, g in pq_params:
        sched = PriorityQueueScheduler(n_sites=20, budget_per_step=3,
                                       alpha=a, beta=b, gamma=g)
        res = evaluator.run(sched)
        row = res.summary()
        row["param_name"] = f"α={a},β={b},γ={g}"
        row["param_value"] = a
        row["algorithm_group"] = "PriorityQueue"
        all_rows.append(row)

    # 2d. Poisson: crawl_cost duyarlılığı
    print("  2d. Poisson MLE - crawl_cost...")
    costs = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    poisson_results = []
    for c in costs:
        sched = PoissonMLEScheduler(n_sites=20, budget_per_step=3, crawl_cost=c)
        res = evaluator.run(sched)
        poisson_results.append((c, res))
        row = res.summary()
        row["param_name"] = "crawl_cost"
        row["param_value"] = c
        row["algorithm_group"] = "Poisson"
        all_rows.append(row)

    # 2e. UCB: exploration_c duyarlılığı
    print("  2e. UCB Bandit - exploration_c...")
    c_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ucb_results = []
    for c in c_vals:
        sched = UCBBanditScheduler(n_sites=20, budget_per_step=3, exploration_c=c)
        res = evaluator.run(sched)
        ucb_results.append((c, res))
        row = res.summary()
        row["param_name"] = "exploration_c"
        row["param_value"] = c
        row["algorithm_group"] = "UCB"
        all_rows.append(row)

    df_params = pd.DataFrame(all_rows)
    df_params.to_csv(RESULTS_DIR / "exp2_sensitivity.csv", index=False)

    _plot_exp2(adapt_results, poisson_results, ucb_results, intervals, costs, c_vals)
    return df_params


def _plot_exp2(adapt_res, poisson_res, ucb_res, intervals, costs, c_vals):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    # Adaptive - base_interval → DFR
    vals = [r.data_freshness_rate for _, r in adapt_res]
    axes[0].plot(intervals, vals, "o-", color="#3498db", linewidth=2, markersize=7)
    axes[0].set_xlabel("base_interval (saat)")
    axes[0].set_ylabel("Data Freshness Rate")
    axes[0].set_title("Adaptive: base_interval\nvs Veri Güncelliği", fontweight="bold")
    axes[0].grid(alpha=0.3)

    # Adaptive - base_interval → Wasted
    vals2 = [r.wasted_crawl_ratio for _, r in adapt_res]
    axes[1].plot(intervals, vals2, "s-", color="#e74c3c", linewidth=2, markersize=7)
    axes[1].set_xlabel("base_interval (saat)")
    axes[1].set_ylabel("Wasted Crawl Ratio")
    axes[1].set_title("Adaptive: base_interval\nvs Boş Tarama Oranı", fontweight="bold")
    axes[1].grid(alpha=0.3)

    # Poisson - crawl_cost → Avg Staleness
    vals3 = [r.avg_staleness for _, r in poisson_res]
    axes[2].plot(costs, vals3, "^-", color="#2ecc71", linewidth=2, markersize=7)
    axes[2].set_xlabel("crawl_cost")
    axes[2].set_ylabel("Avg Staleness")
    axes[2].set_title("Poisson MLE: crawl_cost\nvs Güncellik Kaybı", fontweight="bold")
    axes[2].grid(alpha=0.3)

    # UCB - c → CDR
    vals4 = [r.change_detection_rate for _, r in ucb_res]
    axes[3].plot(c_vals, vals4, "D-", color="#9b59b6", linewidth=2, markersize=7)
    axes[3].set_xlabel("exploration_c")
    axes[3].set_ylabel("Change Detection Rate")
    axes[3].set_title("UCB Bandit: exploration_c\nvs Değişim Yakalama", fontweight="bold")
    axes[3].grid(alpha=0.3)

    # UCB - c → Crawl Efficiency
    vals5 = [r.crawl_efficiency for _, r in ucb_res]
    axes[4].plot(c_vals, vals5, "P-", color="#e67e22", linewidth=2, markersize=7)
    axes[4].set_xlabel("exploration_c")
    axes[4].set_ylabel("Crawl Efficiency (Maliyet/Değişim)")
    axes[4].set_title("UCB Bandit: exploration_c\nvs Tarama Verimliliği", fontweight="bold")
    axes[4].grid(alpha=0.3)

    # Poisson - crawl_cost → CDR
    vals6 = [r.change_detection_rate for _, r in poisson_res]
    axes[5].plot(costs, vals6, "h-", color="#1abc9c", linewidth=2, markersize=7)
    axes[5].set_xlabel("crawl_cost")
    axes[5].set_ylabel("Change Detection Rate")
    axes[5].set_title("Poisson MLE: crawl_cost\nvs Değişim Yakalama", fontweight="bold")
    axes[5].grid(alpha=0.3)

    fig.suptitle("Deney 2: Parametre Duyarlılık Analizi", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp2_sensitivity.png", bbox_inches="tight")
    plt.close()
    print("  → exp2_sensitivity.png kaydedildi")


# ══════════════════════════════════════════════════════════════
# DENEY 3: Bütçe Kısıtı Analizi
# ══════════════════════════════════════════════════════════════

def experiment_3_budget_analysis():
    print("\n" + "="*60)
    print("DENEY 3: Bütçe Kısıtı Altında Performans")
    print("="*60)

    budgets = [1, 2, 3, 5, 8, 10]
    env = ECommerceEnvironment(n_sites=20, seed=42)
    evaluator = Evaluator(env, n_steps=500, time_step=0.5)

    all_rows = []
    for budget in budgets:
        print(f"  Bütçe={budget}...")
        schedulers = [
            PeriodicScheduler(n_sites=20, budget_per_step=budget),
            AdaptiveCrawler(n_sites=20, budget_per_step=budget),
            PriorityQueueScheduler(n_sites=20, budget_per_step=budget),
            PoissonMLEScheduler(n_sites=20, budget_per_step=budget),
            UCBBanditScheduler(n_sites=20, budget_per_step=budget),
        ]
        for sched in schedulers:
            res = evaluator.run(sched)
            row = res.summary()
            row["budget"] = budget
            all_rows.append(row)

    df_budget = pd.DataFrame(all_rows)
    df_budget.to_csv(RESULTS_DIR / "exp3_budget.csv", index=False)
    _plot_exp3(df_budget, budgets)
    return df_budget


def _plot_exp3(df, budgets):
    algo_names = df["Algorithm"].unique()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("Change Detection Rate", "Değişim Yakalama Oranı"),
        ("Avg Staleness",         "Ortalama Güncellik Kaybı"),
        ("Crawl Efficiency",      "Tarama Verimliliği"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for algo in algo_names:
            sub = df[df["Algorithm"] == algo].sort_values("budget")
            ax.plot(sub["budget"], sub[metric], "o-",
                    label=algo, color=get_color(algo), linewidth=2, markersize=6)
        ax.set_xlabel("Bütçe (adım başına tarama)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Bütçe vs {ylabel}", fontweight="bold")
        ax.legend(fontsize=7.5, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_xticks(budgets)

    fig.suptitle("Deney 3: Farklı Bütçe Kısıtları Altında Performans",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp3_budget.png", bbox_inches="tight")
    plt.close()
    print("  → exp3_budget.png kaydedildi")


# ══════════════════════════════════════════════════════════════
# DENEY 4: Ölçeklenebilirlik Analizi
# ══════════════════════════════════════════════════════════════

def experiment_4_scalability():
    print("\n" + "="*60)
    print("DENEY 4: Ölçeklenebilirlik (Site Sayısı)")
    print("="*60)

    site_counts = [5, 10, 20, 50, 100]
    all_rows = []

    for n in site_counts:
        print(f"  n_sites={n}...")
        env = ECommerceEnvironment(n_sites=n, seed=42)
        budget = max(1, n // 7)
        evaluator = Evaluator(env, n_steps=300, time_step=0.5)

        schedulers = [
            PeriodicScheduler(n_sites=n, budget_per_step=budget),
            AdaptiveCrawler(n_sites=n, budget_per_step=budget),
            PriorityQueueScheduler(n_sites=n, budget_per_step=budget),
            PoissonMLEScheduler(n_sites=n, budget_per_step=budget),
            UCBBanditScheduler(n_sites=n, budget_per_step=budget),
        ]
        for sched in schedulers:
            res = evaluator.run(sched)
            row = res.summary()
            row["n_sites"] = n
            row["budget"] = budget
            all_rows.append(row)

    df_scale = pd.DataFrame(all_rows)
    df_scale.to_csv(RESULTS_DIR / "exp4_scalability.csv", index=False)
    _plot_exp4(df_scale, site_counts)
    return df_scale


def _plot_exp4(df, site_counts):
    algo_names = df["Algorithm"].unique()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("Change Detection Rate", "Değişim Yakalama Oranı"),
        ("Wall Clock Time (s)",   "Gerçek Çalışma Süresi (s)"),
        ("Avg Staleness",         "Ortalama Güncellik Kaybı"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for algo in algo_names:
            sub = df[df["Algorithm"] == algo].sort_values("n_sites")
            ax.plot(sub["n_sites"], sub[metric], "o-",
                    label=algo, color=get_color(algo), linewidth=2, markersize=6)
        ax.set_xlabel("Site Sayısı")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Site Sayısı vs {ylabel}", fontweight="bold")
        ax.legend(fontsize=7.5, framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_xticks(site_counts)

    fig.suptitle("Deney 4: Ölçeklenebilirlik Analizi",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp4_scalability.png", bbox_inches="tight")
    plt.close()
    print("  → exp4_scalability.png kaydedildi")


# ══════════════════════════════════════════════════════════════
# DENEY 5: Radar (Spider) Grafiği - Genel Karşılaştırma
# ══════════════════════════════════════════════════════════════

def experiment_5_radar(results_exp1):
    print("\n" + "="*60)
    print("DENEY 5: Radar Grafiği - Çok Boyutlu Karşılaştırma")
    print("="*60)

    results, df = results_exp1
    metrics_radar = [
        "Data Freshness Rate",
        "Change Detection Rate",
        "Wasted Crawl Ratio",  # ters çevrilecek
        "Avg Staleness",       # ters çevrilecek
        "Crawl Efficiency",    # ters çevrilecek
    ]
    labels = [
        "Veri\nGüncelliği",
        "Değişim\nYakalama",
        "Boş\nTarama\n(↓iyi)",
        "Güncellik\nKaybı\n(↓iyi)",
        "Verimlilik\n(↓iyi)",
    ]
    invert = [False, False, True, True, True]

    # Normalize [0,1]
    norm_df = df[metrics_radar].copy()
    for col in metrics_radar:
        mn, mx = norm_df[col].min(), norm_df[col].max()
        if mx > mn:
            norm_df[col] = (norm_df[col] - mn) / (mx - mn)
        else:
            norm_df[col] = 0.5
    for col, inv in zip(metrics_radar, invert):
        if inv:
            norm_df[col] = 1 - norm_df[col]

    N = len(metrics_radar)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for i, res in enumerate(results):
        vals = norm_df.iloc[i].tolist()
        vals += vals[:1]
        color = get_color(res.algorithm_name)
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=res.algorithm_name)
        ax.fill(angles, vals, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.set_title("Algoritma Performans Profilleri\n(Normalize edilmiş — dışarı = daha iyi)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "exp5_radar.png", bbox_inches="tight")
    plt.close()
    print("  → exp5_radar.png kaydedildi")


# ══════════════════════════════════════════════════════════════
# ANA ÇALIŞTIRICI
# ══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Veri Tarama Zamanlama Optimizasyonu - Deney Koşucusu   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    res1 = experiment_1_baseline_comparison()
    df2  = experiment_2_parameter_sensitivity()
    df3  = experiment_3_budget_analysis()
    df4  = experiment_4_scalability()
    experiment_5_radar(res1)

    print("\n" + "="*60)
    print("✅ Tüm deneyler tamamlandı!")
    print(f"   Sonuçlar: {RESULTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()