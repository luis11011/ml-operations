def df__cols__drop_with_single(df, cols="*"):
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