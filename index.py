
def df__cols__drop_mono(df, cols="*"):
    cols = df.cols.names(cols)
    
    if isinstance(cols, str):
        cols = [cols]
        
    from optimus.engines.base.meta import Meta
    
    mono_cols = []
    
    if Meta.get(df.meta, "profile.columns") is not None:
        mono_cols = [col for col in cols if Meta.get(df.meta, f"profile.columns.{col}.stats.count_uniques") == 1]
    
    if len(mono_cols) == 0:
        count_uniques = df.cols.count_uniques(cols, tidy=False)["count_uniques"]
        mono_cols = [col for col, n in count_uniques.items() if n==1]
        
    if mono_cols and len(mono_cols):
        return df.cols.drop(mono_cols)
    
    return df

def df__rows__oversample(_df, features_names, label_name, technique="random"):
    
    if isinstance(features_names, str):
        features_names = [features_names]
    
    if isinstance(label_name, str):
        label_name = [label_name]
        
    _df = _df.cols.select(features_names+label_name).cols.to_float()
        
    X = _df.cols.select(features_names)._to_values()
    y = _df.cols.select(label_name)._to_values()
    
    if technique == "random":
        from imblearn.over_sampling import RandomOverSampler
        over_sampler = RandomOverSampler(random_state=0)
    elif technique == "smote":
        from imblearn.over_sampling import SMOTE
        over_sampler = SMOTE()
    elif technique == "adasyn":
        from imblearn.over_sampling import ADASYN
        over_sampler = ADASYN()
    else:
        raise ValueError(f"technique must be \'random\', \'smote\' or \'adasyn\', received \'{technique}\'")
        
    X_resampled, y_resampled = over_sampler.fit_resample(X, y)
    y_resampled.reshape(-1, 1)
    
    import numpy as np
    import pandas as pd
    
    data = np.concatenate((X_resampled, y_resampled.reshape(-1, 1)), axis=1)
    pdf = pd.DataFrame(data=data, columns=features_names+label_name)
    
    return _df.new(pdf)

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
