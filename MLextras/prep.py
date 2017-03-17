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
