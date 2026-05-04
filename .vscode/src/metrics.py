"""
Değerlendirme Metrikleri
=========================
Her algoritma için hesaplanan metrikler:

1. Data Freshness Rate (DFR)     - Güncel tutulan fiyatların oranı
2. Crawl Efficiency (CE)         - Değişim başına tarama maliyeti
3. Total Resource Cost (TRC)     - Toplam yanıt süresi (saniye cinsinden)
4. Change Detection Rate (CDR)   - Gerçek değişimlerin kaçının yakalandığı
5. Wasted Crawl Ratio (WCR)      - Değişimsiz tarama oranı (israf)
6. Average Staleness (AS)        - Ortalama güncellik kaybı skoru
7. Throughput                    - Birim sürede taranan site sayısı
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time


@dataclass
class CrawlRecord:
    """Tek bir tarama işleminin kaydı."""
    step: int
    time: float
    site_id: int
    price_changed: bool
    response_time: float
    cost: float
    missed_changes: int


@dataclass
class ExperimentResult:
    """Tek bir deney koşumunun sonuçları."""
    algorithm_name: str
    records: List[CrawlRecord] = field(default_factory=list)
    staleness_history: List[float] = field(default_factory=list)
    wall_clock_time: float = 0.0  # Gerçek çalışma süresi

    # Hesaplanan metrikler
    data_freshness_rate: float = 0.0
    crawl_efficiency: float = 0.0
    total_resource_cost: float = 0.0
    change_detection_rate: float = 0.0
    wasted_crawl_ratio: float = 0.0
    avg_staleness: float = 0.0
    throughput: float = 0.0
    total_crawls: int = 0
    total_changes_detected: int = 0

    def compute_metrics(self, total_actual_changes: Dict[int, int]):
        """Tüm metrikleri hesapla."""
        if not self.records:
            return

        self.total_crawls            = len(self.records)
        self.total_changes_detected  = sum(1 for r in self.records if r.price_changed)
        self.total_resource_cost     = sum(r.cost for r in self.records)
        total_missed                 = sum(r.missed_changes for r in self.records)

        # 1. Data Freshness Rate: kaç taramada güncel veri yakalandı
        self.data_freshness_rate = (
            self.total_changes_detected / self.total_crawls
            if self.total_crawls > 0 else 0
        )

        # 2. Crawl Efficiency: değişim başına harcanan maliyet
        self.crawl_efficiency = (
            self.total_resource_cost / max(1, self.total_changes_detected)
        )

        # 3. Change Detection Rate: gerçek değişimlerin kaçını yakaladık
        total_actual = sum(total_actual_changes.values())
        self.change_detection_rate = (
            self.total_changes_detected / max(1, total_actual)
        )
        self.change_detection_rate = min(1.0, self.change_detection_rate)

        # 4. Wasted Crawl Ratio: boş tarama oranı
        wasted = sum(1 for r in self.records if not r.price_changed)
        self.wasted_crawl_ratio = wasted / max(1, self.total_crawls)

        # 5. Average Staleness
        self.avg_staleness = (
            np.mean(self.staleness_history) if self.staleness_history else 0
        )

        # 6. Throughput: birim maliyet başına taranan site
        total_time = self.records[-1].time - self.records[0].time if len(self.records) > 1 else 1
        self.throughput = self.total_crawls / max(1, total_time)

    def summary(self) -> dict:
        return {
            "Algorithm":              self.algorithm_name,
            "Total Crawls":           self.total_crawls,
            "Changes Detected":       self.total_changes_detected,
            "Data Freshness Rate":    round(self.data_freshness_rate, 4),
            "Crawl Efficiency":       round(self.crawl_efficiency, 4),
            "Total Resource Cost":    round(self.total_resource_cost, 2),
            "Change Detection Rate":  round(self.change_detection_rate, 4),
            "Wasted Crawl Ratio":     round(self.wasted_crawl_ratio, 4),
            "Avg Staleness":          round(self.avg_staleness, 4),
            "Throughput":             round(self.throughput, 4),
            "Wall Clock Time (s)":    round(self.wall_clock_time, 3),
        }


class Evaluator:
    """Algoritma değerlendirme çerçevesi."""

    def __init__(self, env, n_steps: int = 500, time_step: float = 0.5):
        self.env = env
        self.n_steps = n_steps
        self.time_step = time_step

    def run(self, scheduler, verbose: bool = False) -> ExperimentResult:
        """
        Bir zamanlayıcıyı simülasyon ortamında çalıştır.
        
        Her adımda:
        1. Zamanlayıcıdan site seçimi al
        2. Seçilen siteleri tara
        3. Sonuçları zamanlayıcıya bildir
        4. Metrikleri kaydet
        """
        self.env.reset()
        scheduler.reset()
        result = ExperimentResult(algorithm_name=scheduler.name)

        start_wall = time.perf_counter()

        for step in range(self.n_steps):
            current_time = step * self.time_step
            sites_to_crawl = scheduler.select_sites(current_time)

            for site_id in sites_to_crawl:
                crawl_result = self.env.crawl(site_id, current_time)
                scheduler.update(site_id, crawl_result, current_time)

                result.records.append(CrawlRecord(
                    step=step,
                    time=current_time,
                    site_id=site_id,
                    price_changed=crawl_result["price_changed"],
                    response_time=crawl_result["response_time"],
                    cost=crawl_result["cost"],
                    missed_changes=crawl_result["missed_changes"],
                ))

            # Her adımda toplam staleness'ı kaydet
            staleness = self.env.total_staleness(current_time)
            result.staleness_history.append(staleness)

            if verbose and step % 100 == 0:
                print(f"  Step {step:4d} | time={current_time:.1f} | "
                      f"staleness={staleness:.2f} | crawls={len(result.records)}")

        result.wall_clock_time = time.perf_counter() - start_wall

        # Gerçek değişim sayılarını env'den al
        total_actual = {s.site_id: s.change_count for s in self.env.sites}
        result.compute_metrics(total_actual)

        return result

    def run_with_params(self, scheduler_class, param_grid: list,
                        base_kwargs: dict = None) -> List[ExperimentResult]:
        """Parametre tarama deneyleri."""
        base_kwargs = base_kwargs or {}
        results = []
        for params in param_grid:
            kwargs = {**base_kwargs, **params}
            sched = scheduler_class(**kwargs)
            res = self.run(sched)
            res.algorithm_name = f"{sched.name} | {params}"
            results.append(res)
        return results