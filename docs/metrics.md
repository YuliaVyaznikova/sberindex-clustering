# Метрики качества кластеров

Код в `src/metrics.py`.

| Метрика | Пространство | Лучше | Источник |
|---|---|---|---|
| SW, Silhouette Width | признаки | больше | Rousseeuw 1987 |
| CH, индекс Calinski-Harabasz | признаки | больше | Caliński, Harabasz 1974 |
| S_Dbw = Scat + Dens_bw | признаки | меньше | Halkidi, Vazirgiannis 2001 |
| AVI, Average Isolability | граф | больше | Biswas, Biswas 2017 |
| AVU, Average Unifiability | граф | меньше | Biswas, Biswas 2017 |
| MQ, модулярность Q | граф | больше | Newman, Girvan 2004 |

Дополнительно считаются индекс Davies-Bouldin, ANUI = AVI / (1 + AVI * AVU), density modularity (Kim et al. 2022) и доля веса рёбер внутри кластеров `within` (сумма S_ii, делённая на сумму всех S_ij): какая часть связей похожих по потреблению МО не разрезана границами кластеров. MQ в критериях конкурса не расшифрована, мы считаем её модулярностью Q.

Определения:
- S_ij это суммарный вес рёбер между кластерами i и j, out_i это сумма S_ij по всем j, кроме i.
- AVI это среднее по кластерам S_ii / (S_ii + out_i).
- AVU это среднее по парам кластеров S_ij / (out_i + out_j - S_ij).
- В S_Dbw разброс кластера берётся как вектор дисперсий, как в статье.

Значения графовых метрик зависят от числа кластеров k даже у случайного разбиения: у него AVI около 1/k, а AVU ровно 1/(2k - 3) при равномерном перемешивании (при k = 3 AVU равна 1/3 для любого разбиения). Поэтому для каждого разбиения считается и случайный уровень, среднее по 10 перестановкам меток с теми же размерами кластеров, и разница с ним (`AVI_lift`, `AVU_gain`, `MQ_lift`). Устойчивость разбиения считается как средний ARI между разбиением всех МО и разбиениями случайных 90% МО (`compare.subsamples` подвыборок, одни и те же для всех методов). Для совместных методов она считается при alpha из `compare.stable_alphas`. Пять подвыборок дают грубую оценку: у одного метода ARI на отдельных подвыборках разбросан на 0.1-0.2.

## Литература

- Rousseeuw P. J. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53-65, 1987.
- Caliński T., Harabasz J. A dendrite method for cluster analysis. Communications in Statistics, 3(1), 1-27, 1974.
- Davies D. L., Bouldin D. W. A cluster separation measure. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1(2), 224-227, 1979.
- Halkidi M., Vazirgiannis M. Clustering validity assessment: finding the optimal partitioning of a data set. Proceedings of IEEE ICDM, 187-194, 2001.
- Biswas A., Biswas B. Defining quality metrics for graph clustering evaluation. Expert Systems with Applications, 71, 1-17, 2017.
- Howie J. et al. Scaling up structural clustering to large probabilistic graphs. Proceedings of the VLDB Endowment, 16, 2023 (формулы AVI, AVU, ANUI в явном виде).
- Newman M. E. J., Girvan M. Finding and evaluating community structure in networks. Physical Review E, 69, 026113, 2004.
- Kim J., Luo S., Cong G., Yu W. DMCS: Density Modularity based Community Search. Proceedings of SIGMOD, 2022.
- Shalileh S., Antonov E., Tsyplakova D. Internal cluster validity indices for attributed networks: a controlled comparative study. Expert Systems with Applications, doi 10.1016/j.eswa.2026.133912.