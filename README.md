# E-Commerce Crawl Scheduler Optimization

Bu proje, farklı zamanlama (scheduling) algoritmalarını kullanarak e-ticaret sitelerindeki fiyat değişimlerini en verimli şekilde tespit etmeyi amaçlayan bir simülasyon ortamıdır. Sınırlı bütçe/tarama hakkı ile en yüksek güncellik oranını (Data Freshness Rate) yakalamak hedeflenmektedir.

##  Proje Yapısı


* **`environment.py`:** Simülasyon ortamını (`ECommerceEnvironment`) içerir. Sitelerin değişim olasılıklarını (Poisson dağılımı vb.), fiyat dinamiklerini ve yanıt sürelerini simüle eder.
* **`algorithms.py`:** Tarama zamanlama algoritmalarını içerir (Adaptive Crawling, Priority Queue, Poisson MLE, UCB Bandit, DQN vb.).
* **`metrics.py`:** Değerlendirme altyapısını (`Evaluator`) ve metrik hesaplamalarını (Güncellik oranı, Boş tarama oranı, Verimlilik vb.) içerir.
* **`experiments.py`:** Ana deney koşucusudur. Çeşitli senaryolarda algoritmaları karşılaştırır, tablolar oluşturur ve grafikleri çizer.

##  Kurulum

Projeyi çalıştırmak için Python 3.7+ ve aşağıdaki kütüphanelerin yükle:

```bash
pip install numpy pandas matplotlib seaborn
