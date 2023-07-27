import numpy as np
import pandas as pd

class Node:
    def __init__(self,
                 feature = None,
                 feature_value = None,
                 threshold = None,
                 data_left = None,
                 data_right = None,
                 gain = None,
                 value = None):
        
        self.feature = feature
        self.feature_value = feature_value
        self.threshold = threshold
        self.data_left = data_left 
        self.data_right = data_right
        self.gain = gain
        self.value = value
        
class DecisionTree:
    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def entropy(self, data):
        #Total class counts
        class_counts = np.bincount(data)
        class_probs = class_counts/len(data)
        
        
        
        entropy = 0
        class_entropies = []
        for prob in class_probs:
            if prob > 0:
                entropy += prob * np.log(prob)
                class_entropies.append(entropy)
        entropy = np.sum(class_entropies) * -1
        return entropy
        
        
        
        
    def information_gain(self, parent, left_child, right_child):
        
        
        num_left = len(left_child)/len(parent)
        num_right = len(right_child)/len(parent)
        
        #compute the entropies
        parent_entropy = self.entropy(parent)
        left_entropy = self.entropy(left_child)
        right_entropy = self.entropy(right_child)
        
        #compute information gain, aggregating the entropies
        info_gain = parent_entropy - (num_left * left_entropy + num_right * right_entropy)
        
        return info_gain
    
    def _best_split(self, X, y):
        
        best_gain = -1
        best_split = {}
        
        #loop through each dataset feature
        df = np.concatenate((X, np.array(y).reshape(1, -1).T), axis = 1)
        df = pd.DataFrame(df)
        
        feature_set = list(X.columns)
        
        for feature_col in feature_set:
            #store the feature data somewhere
            feature_data = sorted(np.unique(X[feature_col]))
            
            for feature_val in feature_data:
                df_left = df[df[featue] <= feature.val].copy()
                df_right = df[df[feature] > feature.val].copy()
        
                y_parent = df['y']
                y_left = df_left['y']
                y_right = df_right['y']
        
                #compute Information gain
                info_gain = self.information_gain(y_parent, y_left, y_right)
                
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_split = {
                        'feature_col': feature_col,
                        'split_value': feature_val,
                        'df_left': df_left,
                        'df_right': df_right,
                        'gain': info_gain
                        }
        return best_split
        
    
    
    
    def _build_tree(self, X, y, depth):
        
        
    
        best = self.best_split(x, y)    
    
        if n_rows >= self.min_samples_split and depth <= self.max_depth:
            ## left
            left = self._build_tree(
                X = best_df['df_left'].drop(['y'], axis = 1),
                y = best_df['y'],
                depth = depth + 1
                )
        
            ## right
            right = self._build.tree(
                X = best_df['df_right'].drop(['y'], axis = 1),
                y = best_df['y'],
                depth = depth + 1
                )
    
            return Node(feature = best['feature_col'],
                        threshold = best['split_value'],
                        data_left = left,
                        data_right = right,
                        gain = best['gain'])
        
        # return the node to our tree
        return Node(value = Counter(y).most_common(1)[0][0])
    
        
        
    def fit(self):
        # Build the tree
        # - Within a tree, we need to be able to find the best splits        
        self.root = self._build_tree(X, y)
    
    def _traverse_tree(self, x, node):
        # if we hit the laf node, return that value
        if node.value != None:
            return node.value
        
        #pull feature column from the node
        feature_value = x[node.feature]
        
        # go left if less than threshold
        if feature_value <= node.threshold:
            return self._traverse_tree(x = x, node = node.data_left)
        
        #go right if more than threshold
        if feature_value > node.threshold:
            return self._traverse_tree(x = x, node = node.data_right)
        
    def predict(self, X):
        #traverse the tree to make a prediction
        predictions = []
        for index, x in X.iterrows():
            pred = self._traverse_tree(x, self.root)
            predictions.append(pred)
        return predictions