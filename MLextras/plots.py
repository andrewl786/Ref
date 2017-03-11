def plot_learning_curve(clf, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    return plt

def visualizetree(clf):
    dot_data = export_graphviz(clf, out_file=None, feature_names=list(X_train.columns),
                               class_names=True,filled=True, rounded=True,
                               proportion=True,
                               special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    display(Image(graph.create_png()))

def visualizeimportance(clf, title, figsize):
    featureimp = pd.Series(clf.feature_importances_, X_all.columns).sort_values(ascending=False)
    featureimp = featureimp[featureimp > 0].sort_values(ascending=True)
    featureimp.plot(kind='barh', title=title, figsize=figsize);
    plt.xlabel('Gini importance');

def visualizeimportance_horizontal(clf, title, figsize):
    featureimp = pd.Series(clf.feature_importances_, X_all.columns).sort_values(ascending=False)
    featureimp = featureimp[featureimp > 0].sort_values(ascending=False)
    featureimp.plot(kind='bar', title=title, figsize=figsize);
    plt.ylabel('Gini importance');
