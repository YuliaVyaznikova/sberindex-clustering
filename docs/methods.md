# Методы кластеризации

Код в `src/methods.py`, `src/kefrin.py` и `src/canus.py`, сравнение в `src/compare.py`, параметры в разделе `compare` файла `config.yaml`.

- Только признаки: k-means, иерархическая кластеризация Уорда, смесь гауссиан.
- Только граф: спектральная кластеризация, Louvain.
- Признаки и граф вместе:
  - KEFRiN (Shalileh, Mirkin 2022), K-Means Extended to Feature-Rich Networks, своя реализация по статье. У кластера два центра: в пространстве признаков и в пространстве строк матрицы связей. Узел относится к кластеру с наименьшим rho * d(x_i, c_k) + xi * d(p_i, lambda_k). Граф стандартизуется как p_ij - p_i+ p_+j / p_++. Варианты с евклидовым и косинусным расстоянием.
  - CANUS (Shalileh 2025), Clustering Attributed Network Using the Steepest descent, своя реализация варианта с фильтром (CANUSf) по статье. Критерий тот же, что у KEFRiN, но центры сдвигаются градиентным шагом после каждого узла. Сначала 100 проходов по бутстреп-выборкам узлов дают среднее mu и разброс sigma нормы градиента, затем в 10 проходах центр сдвигается, только если норма градиента узла лежит в mu ± tau * sigma. Значения из статьи: шаг 0.1, tau = 0.3, старт из K случайных узлов, один запуск. Признаки и стандартизованный граф делятся на средний разброс на узел, rho = 1 - alpha, xi = alpha. Варианты с евклидовым и косинусным расстоянием; в косинусном строки и центры нормируются на единицу.
  - Совместная спектральная кластеризация: матрица сродства alpha * W_граф + (1 - alpha) * W_признаки, где W_признаки это граф 10 ближайших соседей по всем признакам, обе матрицы нормированы на сумму весов.

Вес графа задаётся одним параметром alpha от 0 до 1: для KEFRiN rho = (1 - alpha) / T_x и xi = alpha / T_p, где T_x и T_p это полный разброс признаков и стандартизованного графа. alpha = 0 даёт кластеризацию только по признакам, alpha = 1 только по графу. Сравнение проходит по всей сетке alpha.

## Литература

- Shalileh S., Mirkin B. Community partitioning over feature-rich networks using an extended k-means method. Entropy, 24(5), 626, 2022.
- Shalileh S. A filtered gradient descent clustering method to recover communities in attributed networks. IEEE Access, 13, 2025. https://doi.org/10.1109/ACCESS.2025.3614989
- Damle A., Minden V., Ying L. Simple, direct and efficient multi-way spectral clustering. Information and Inference, 8(1), 181-203, 2019.
- Blondel V. D., Guillaume J.-L., Lambiotte R., Lefebvre E. Fast unfolding of communities in large networks. Journal of Statistical Mechanics, P10008, 2008.
- Ward J. H. Hierarchical grouping to optimize an objective function. Journal of the American Statistical Association, 58(301), 236-244, 1963.
