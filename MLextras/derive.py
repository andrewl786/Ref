# data prep and feature derivations: calculate mean, min, max for each column
def aggflatten(df):
    cols = df.columns
    agg_funcs = ['mean', 'min', 'max']

    # group by unique identifiers and aggregate data
    g = df.groupby(['subject_id', 'hadm_id', 'icustay_id']).agg(agg_funcs)

    # concatenate variable with aggregation function, remove concatenated
    # identifiers
    new_cols = [x + '_' + y for x in cols for y in agg_funcs]

    # drop old column hierarchy, set concatenated column names
    g.columns = g.columns.droplevel(0)
    g.columns = new_cols

    return g.reset_index()
