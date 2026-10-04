# Методы кластеризации

Код в `src/methods.py`, `src/kefrin.py`, `src/canus.py` и `src/dmon.py`, сравнение в `src/compare.py`, параметры в разделе `compare` файла `config.yaml`.

- Только признаки: k-means, иерархическая кластеризация Уорда, GMM.
- Только граф: спектральная кластеризация, Louvain, Leiden. Leiden (Traag et al. 2019) это Louvain с шагом уточнения, который гарантирует связность каждого сообщества; используется реализация networkx с той же целевой функцией (модулярность). У Louvain и Leiden число кластеров задаётся не напрямую, а параметром resolution: перебирается `compare.resolutions` значений от 0.05 до 3 и для каждого k берётся первое значение resolution, которое даёт k сообществ.
- Признаки и граф вместе:
  - KEFRiN (Shalileh, Mirkin 2022), K-Means Extended to Feature-Rich Networks, своя реализация по статье. У кластера два центра: в пространстве признаков и в пространстве строк матрицы связей. Узел относится к кластеру с наименьшим rho * d(x_i, c_k) + xi * d(p_i, lambda_k). Граф стандартизуется как p_ij - p_i+ p_+j / p_++. Варианты с евклидовым и косинусным расстоянием.
  - CANUS (Shalileh 2025), Clustering Attributed Network Using the Steepest descent, своя реализация варианта с фильтром (CANUSf) по статье. Критерий тот же, что у KEFRiN, но центры сдвигаются градиентным шагом после каждого узла. Сначала 100 проходов по бутстреп-выборкам узлов дают среднее mu и разброс sigma нормы градиента, затем в 10 проходах центр сдвигается, только если норма градиента узла лежит в mu ± tau * sigma. Значения из статьи: learning rate 0.1, tau = 0.3, старт из K случайных узлов, один запуск. Признаки и стандартизованный граф делятся на средний разброс на узел, rho = 1 - alpha, xi = alpha. Варианты с евклидовым и косинусным расстоянием; в косинусном строки и центры нормируются на единицу.
  - DMoN (Tsitsulin et al. 2023), Deep Modularity Networks, своя реализация на PyTorch по статье. Однослойная GCN со skip connection вместо петель: H = SeLU(A_n X W + X W_skip), где A_n = D^(-1/2) A D^(-1/2). Soft assignment узлов C = softmax(dropout(H) W_c + b). Функция потерь -Tr(C^T B C) / 2m + lambda * (sqrt(k) / n * ||сумма строк C|| - 1): первая часть это модулярность при soft assignment (B матрица модулярности), вторая (collapse regularization) не даёт всем узлам уйти в один кластер. Узел относится к кластеру с наибольшим значением в своей строке C. Из статьи: 64 нейрона в скрытом слое (как в опытах авторов на небольших графах) и dropout 0.5. Статья не называет learning rate и число эпох, они взяты из значений по умолчанию в открытой реализации авторов: Adam с learning rate 0.001, 1000 эпох, lambda = 1. Граф DMoN это та же смесь alpha * W_граф + (1 - alpha) * W_признаки, что у совместной спектральной кластеризации, признаки узлов это x_i. Обучение идёт в два потока: на графе из 1800 узлов больше потоков только замедляет его.
  - Совместная спектральная кластеризация: affinity matrix alpha * W_граф + (1 - alpha) * W_признаки, где W_признаки это граф 10 ближайших соседей по всем признакам, обе матрицы нормированы на сумму весов.

Вес графа задаётся одним параметром alpha от 0 до 1: для KEFRiN rho = (1 - alpha) / T_x и xi = alpha / T_p, где T_x и T_p это полный разброс признаков и стандартизованного графа. alpha = 0 даёт кластеризацию только по признакам, alpha = 1 только по графу. Сравнение проходит по всей сетке alpha.

## Литература

- Shalileh S., Mirkin B. Community partitioning over feature-rich networks using an extended k-means method. Entropy, 24(5), 626, 2022.
- Shalileh S. A filtered gradient descent clustering method to recover communities in attributed networks. IEEE Access, 13, 2025. https://doi.org/10.1109/ACCESS.2025.3614989
- Damle A., Minden V., Ying L. Simple, direct and efficient multi-way spectral clustering. Information and Inference, 8(1), 181-203, 2019.
- Tsitsulin A., Palowitch J., Perozzi B., Müller E. Graph clustering with graph neural networks. Journal of Machine Learning Research, 24, 1-21, 2023.
- Klambauer G., Unterthiner T., Mayr A., Hochreiter S. Self-normalizing neural networks. Advances in Neural Information Processing Systems 30, 2017.
- Blondel V. D., Guillaume J.-L., Lambiotte R., Lefebvre E. Fast unfolding of communities in large networks. Journal of Statistical Mechanics, P10008, 2008.
- Traag V. A., Waltman L., van Eck N. J. From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports, 9, 5233, 2019.
- Ward J. H. Hierarchical grouping to optimize an objective function. Journal of the American Statistical Association, 58(301), 236-244, 1963.