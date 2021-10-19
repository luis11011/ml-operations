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