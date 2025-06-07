import pandas as pd
import numpy as np
from FindFastUShapelet import FindFastUShapelet
from GetActualGap import GetActualGap


def load_dataset(path):
    """Load daily generation data from CSV.

    The CSV is expected to have dates as the first column and time
    points as the remaining columns.
    """
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(float)


def cluster_daily_data(path, shapelet_length=30):
    data = load_dataset(path)
    # use dummy labels since the algorithm expects them
    class_labels = np.zeros(data.shape[0])

    remaining_ind = np.arange(data.shape[0])
    labels_result = np.zeros((data.shape[0], 1))
    current_cluster = 1
    min_gap = 0

    while len(remaining_ind) > 3:
        best_idx, best_shapelets, s_len, cls_num, _, _, _, _ = FindFastUShapelet(
            data, class_labels, path, shapelet_length
        )
        max_gap = best_shapelets[best_idx, 2]
        _, _, new_idx = GetActualGap(s_len, best_shapelets, best_idx, data,
                                     class_labels, cls_num)

        ts = remaining_ind[int(best_shapelets[best_idx, 0])]
        bsf_current_idx = new_idx

        if min_gap == 0:
            if max_gap > 0:
                min_gap = max_gap
            else:
                break
        else:
            if min_gap / 2 > max_gap:
                break

        ind_to_delete = np.argwhere(bsf_current_idx)
        data = np.delete(data, np.concatenate(ind_to_delete), axis=0)
        labels_result[remaining_ind[ind_to_delete]] = current_cluster
        remaining_ind = np.delete(remaining_ind, np.concatenate(ind_to_delete),
                                  axis=0)
        current_cluster += 1

    return labels_result.flatten()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cluster daily generation data without labels")
    parser.add_argument("csv_file", help="CSV file containing the data")
    parser.add_argument("--shapelet_length", type=int, default=30,
                        help="Length of shapelets to use")
    args = parser.parse_args()

    labels = cluster_daily_data(args.csv_file, args.shapelet_length)
    print("Cluster labels:", labels)
