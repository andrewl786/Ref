# create specificity scorer
def specificity(y, y_pred):
    # calculates using confusiong matrix
    cm = confusion_matrix(y, y_pred)
    res = cm[0,0]*1.0 / np.sum(cm[0,:])
    return res

# create method that will pass a list of classifiers through scoring metrics and yield a results table
def scoremetrics(clfs):

    results_list = []

    for clf in clfs:
        if str(clf).split('(',1)[0] == 'GradientBoostingClassifier':
            # fit GB with balanced samples
            clf.fit(X_train, y_train, sample_weight=(np.ones(X_train.shape[0]) / X_train.shape[0]))
        else:
            clf.fit(X_train, y_train)

        res = {}
        # extract the model name from the object string
        res['Model'] = str(clf).split('(',1)[0]
        # get parameters from attribute
        res['Parameters'] = clf.get_params()

        # within each classifier, cycle through metric scorers with train and test sets
        for metric in [roc_auc_score, specificity, precision_score, recall_score]:
            metric_name = metric.__name__
            res[metric_name + '_train'] = metric(y_train, clf.predict(X_train))
            res[metric_name + '_test'] = metric(y_test, clf.predict(X_test))

        results_list.append(res)
        results_df = pd.DataFrame(results_list)

    return results_df

# contains column references to MIMIC-III database and has not yet been updated
# for general use
def cumulativelift(clf):
    # combine predicted probability with target
    pred = pd.DataFrame(clf.predict_proba(X_test)[:,1], columns=['propensity'])
    truth = pd.DataFrame(y_test.reset_index(drop=True))
    df = pd.merge(truth, pred, left_index=True, right_index=True)

    # calculate lift per decile of ranked probability
    df.sort(columns='propensity', ascending=False, inplace=True)
    df.reset_index(drop=True)
    df['seq'] = range(1, len(df) + 1)
    df['percentile'] = 100 - np.floor(df['seq'] *100.0 / (df.shape[0])) - 1
    df.ix[df.percentile==-1.0] = 0.0

    # probability of correctly discriminating target with random chance
    randomchance = df.death.sum() / df.death.count()

    # calculate cumulative lift by percentile
    g = df.groupby('percentile', as_index=True).agg(OrderedDict([
                ('death',
                OrderedDict([
                            ('deaths', 'sum'),
                            ('obs', 'count')
                        ]))
            ])).reset_index(col_level=1)
    # flatten headers
    g.columns = g.columns.get_level_values(1)
    # calculate cumulative lift
    g.sort(columns='percentile', ascending=False, inplace=True)
    g['cumulativelift'] = (g.deaths.cumsum() / g.obs.cumsum()) / randomchance

    return g
