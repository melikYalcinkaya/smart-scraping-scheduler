import numpy as np
import random
from dataclasses import dataclass
from typing import List

@dataclass
class SiteProfile:
    site_id: int
    name: str
    true_lambda: float
    base_response_time: float
    response_variance: float
    category: str

@dataclass
class CrawlResult:
    site_id: int
    timestamp: float
    price_changed: bool
    response_time: float
    new_price: float
    crawl_cost: float

class ECommerceEnvironment:
    SITE_CONFIGS = [
        SiteProfile(0,  "FlashDeal",      0.80, 0.8, 0.2, "volatile"),
        SiteProfile(1,  "AuctionHub",     0.70, 1.2, 0.5, "volatile"),
        SiteProfile(2,  "DynamicShop",    0.60, 0.9, 0.3, "volatile"),
        SiteProfile(3,  "SpeedMart",      0.55, 0.6, 0.1, "volatile"),
        SiteProfile(4,  "MegaStore",      0.30, 1.5, 0.4, "moderate"),
        SiteProfile(5,  "TechZone",       0.25, 2.0, 0.6, "moderate"),
        SiteProfile(6,  "HomeShopper",    0.20, 1.8, 0.5, "moderate"),
        SiteProfile(7,  "BestBuy",        0.22, 1.4, 0.3, "moderate"),
        SiteProfile(8,  "LuxuryGoods",    0.05, 3.0, 1.0, "stable"),
        SiteProfile(9,  "SpecialtyStore", 0.08, 2.5, 0.8, "stable"),
        SiteProfile(10, "BookShelf",      0.04, 1.0, 0.2, "stable"),
        SiteProfile(11, "GourmetShop",    0.06, 2.2, 0.7, "stable"),
    ]

    def __init__(self, n_sites=12, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.n_sites = n_sites
        self.sites = self._build_sites(n_sites)
        self.reset()

    def _build_sites(self, n_sites):
        base = list(self.SITE_CONFIGS)
        if n_sites <= len(base):
            return base[:n_sites]

        extra = []
        categories = ["volatile", "moderate", "stable"]
        lambdas = [0.75, 0.45, 0.12]
        response_means = [1.0, 1.8, 2.5]
        response_vars = [0.35, 0.55, 0.85]

        for idx in range(len(base), n_sites):
            cat = categories[idx % len(categories)]
            lam = lambdas[idx % len(lambdas)] * (0.9 + 0.1 * ((idx % 3)))
            rt = response_means[idx % len(response_means)] * (1.0 + 0.05 * ((idx // 3) % 3))
            var = response_vars[idx % len(response_vars)]
            name = f"Site{idx:02d}"
            extra.append(SiteProfile(idx, name, lam, rt, var, cat))

        return base + extra

    def reset(self):
        self.current_time = 0.0
        self.prices = {s.site_id: round(random.uniform(10, 500), 2) for s in self.sites}
        self.last_change_time = {s.site_id: 0.0 for s in self.sites}
        self.change_count = {s.site_id: 0 for s in self.sites}
        self.total_crawls = 0
        self.total_cost = 0.0

    def advance_time(self, delta):
        self.current_time += delta

    def crawl_site(self, site_id):
        site = self.sites[site_id]
        elapsed = self.current_time - self.last_change_time[site_id]
        p_change = 1.0 - np.exp(-site.true_lambda * elapsed)
        price_changed = np.random.random() < p_change
        if price_changed:
            chg = np.random.uniform(0.05, 0.30) * np.random.choice([-1, 1])
            self.prices[site_id] = round(self.prices[site_id] * (1 + chg), 2)
            self.last_change_time[site_id] = self.current_time
        response_time = max(0.1, np.random.lognormal(
            mean=np.log(site.base_response_time), sigma=site.response_variance))
        crawl_cost = response_time / 10.0
        self.total_crawls += 1
        self.total_cost += crawl_cost
        return CrawlResult(site_id, self.current_time, price_changed,
                           response_time, self.prices[site_id], crawl_cost)

    def get_staleness(self, site_id, last_crawl_time):
        elapsed = self.current_time - last_crawl_time
        return 1.0 - np.exp(-self.sites[site_id].true_lambda * elapsed)