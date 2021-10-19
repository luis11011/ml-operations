def df__cols__knn_impute(df, cols="*", n_neighbors=5, weights="uniform", metric="nan_euclidean", output_cols=None):
    def _knn_impute(series, n_neighbors=5, weights="uniform", metric="nan_euclidean"):
        """

        :param series:
        :param missing_values:
        :param n_neighbors:
        :param weights: {\'uniform\', \'distance\'} or callable, default=\'uniform\'
        :param metric: {\'nan_euclidean\'} or callable, default=\'nan_euclidean\'
        :return:
        """
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric)
        return imputer.fit_transform(series.values.reshape(-1, 1))
    
    from optimus.helpers.columns import parse_columns, get_output_cols, prepare_columns_arguments

    cols = parse_columns(df, cols)

    n_neighbors, weights, metric = prepare_columns_arguments(cols, n_neighbors, weights, metric)
    output_cols = get_output_cols(cols, output_cols)

    for col_name, output_col, _n_neighbors, _weights, _metric in zip(cols, output_cols, n_neighbors, weights, metric):

        df = df.cols.to_float(col_name)
        df = df.cols.apply(col_name, _knn_impute, output_cols=output_col, args=(_n_neighbors, _weights, _metric), mode="vectorized")

    return df