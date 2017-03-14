# data prep that replaces 'None' with NaN and creates dummies
def preprocess(df):
    output = pd.DataFrame(index=df.index)

    if 'charttime' in df:
        del df['charttime']

    for col, col_data in df.iteritems():

        # delete date columns
        #if col_data.dtype == np.datetime64:
        #    df.drop(col, axis=1)

        # if data is non-numeric, replace with...
        if col_data.dtype == object:
            col_data = col_data.replace(['None'], [np.NaN])

        # if data is categorical, create dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)

        # collect the revised columns
        output = output.join(col_data)

    return output

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
