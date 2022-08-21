# CONSTANTS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

cat_feats = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]

# PIPELINES
df_cols_nn = ['mean_fit_time',
            'std_fit_time',
            'param_clf__activation', 
            'param_clf__alpha',
            'param_clf__hidden_layer_sizes', 
            'param_clf__max_iter',
            'param_clf__solver',
            'mean_score_time',
            'std_score_time',
            'mean_test_PR-AUC',
            'rank_test_PR-AUC',
            'mean_test_avg-precision',
            'rank_test_avg-precision',
            'mean_test_AUC',
            'rank_test_AUC',
            'mean_test_Accuracy',
            'rank_test_Accuracy',
            'mean_test_prec',
            'rank_test_prec',
            'mean_test_recall',
            'rank_test_recall',
            'mean_test_F1',
            'rank_test_F1']

# Default pipeline steps
preprocessor_mlp = Pipeline(steps=[
    ("preprocessing", StandardScaler()),
])