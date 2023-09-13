from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class CustomDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, max_impurity, default_value, **kwargs):
        super().__init__(**kwargs)
        self.max_impurity = max_impurity
        self.default_value = default_value

    def predict(self, X, check_input=True):
        leaf_indices = self.apply(X)
        node_impurities = self.tree_.impurity[leaf_indices]
        predictions = super().predict(X, check_input)
        predictions[node_impurities > self.max_impurity] = self.default_value
        return predictions

    def predict_proba(self, X, check_input=True):
            leaf_indices = self.apply(X)
            node_impurities = self.tree_.impurity[leaf_indices]
            proba = super().predict_proba(X, check_input)
            proba[node_impurities > self.max_impurity] = [1, 0]  # assuming binary classification
            return proba

class CustomRandomForestClassifier(RandomForestClassifier):
    def __init__(self, n_estimators=1,max_depth=3, min_weight_fraction_leaf=0.1, max_features=2, max_samples=0.65, class_weight='balanced_subsample', max_impurity=0.4, default_value=0, **kwargs):
        super().__init__(n_estimators=n_estimators,max_depth=max_depth, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_samples=max_samples, class_weight=class_weight, **kwargs)
        self.max_impurity = max_impurity
        self.default_value = default_value

    def _make_estimator(self, append=True,random_state=None):
        estimator = CustomDecisionTreeClassifier(max_impurity=self.max_impurity, default_value=self.default_value,random_state=random_state,max_depth=self.max_depth, min_weight_fraction_leaf=self.min_weight_fraction_leaf, max_features=self.max_features, class_weight="balanced")
        if append:
            self.estimators_.append(estimator)
        return estimator
    
