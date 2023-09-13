
from sklearn.tree import export_graphviz,export_text
import os
import glob
def save_trees(wrapper, folder_path):
    features = wrapper.feature_engineer.sampler.sampling_cols.copy()
    # if wrapper.feature_engineer.bucket_return: features += ["bucket_return"]
    for i, tree in enumerate(wrapper.model.estimators_[:10]):
        # print(export_text(tree,feature_names=wrapper.feature_engineer.sampler.cols,class_names=["Not Bet","Bet"]))
        export_graphviz(tree, out_file=f'{folder_path}rf_{i}.dot', 
                        feature_names=features,
                        class_names=["Not Bet","Bet"],
                        rounded=True,
                        proportion=False, 
                        precision=2, 
                        filled=True
        )
        
        # Convert to PNG using system command
        os.system(f'dot -Tpng -Gdpi=300 {folder_path}rf_{i}.dot -o {folder_path}rf_{i}.png')

    interpretation = {
        'net_flow_native' : "Changes of net flow into the bitcoin market",
        'balance_1k_native' : "Changes in number of institutional investors in bitcoin",
        'close' : "Changes in price of bitcoin",
        'balance_1k_usd' : "Changes in number of retail investors in bitcoin",
        'vix_close' : "Changes in market volatility/sentiment",
        'spy_close' : "Changes in Economic conditions",
    }

    for i, tree in enumerate(wrapper.model.estimators_[:10]):
        # print(export_text(tree,feature_names=wrapper.feature_engineer.sampler.cols,class_names=["Not Bet","Bet"]))
        export_graphviz(tree, out_file=f'{folder_path}rf_{i}.dot', 
                        feature_names=[interpretation[f] for f in features],
                        class_names=["Not Bet","Bet"],
                        rounded=True,
                        proportion=False, 
                        precision=2, 
                        filled=True
        )
        
        # Convert to PNG using system command
        os.system(f'dot -Tpng -Gdpi=300 {folder_path}rf_{i}.dot -o {folder_path}interpretation_{i}.png')

    

    # find all files in the folder that end with .dot
    files = glob.glob(os.path.join(folder_path, '*.dot'))

    # iterate over the list of filepaths & remove each file.
    for file in files:
        try:
            os.remove(file)
            print(f'{file} has been removed successfully')
        except Exception as e:
            print(f'Error occurred while deleting {file}. Error message: {str(e)}')


def save_one_tree(model,folder_path,features):
    export_graphviz(model, out_file=f'{folder_path}primary_model.dot', 
                        feature_names=features,
                        class_names=["Short","Long"],
                        rounded=True,
                        proportion=False, 
                        precision=2, 
                        filled=True
        )
        
    # Convert to PNG using system command
    os.system(f'dot -Tpng -Gdpi=300 {folder_path}primary_model.dot -o {folder_path}primary_model.png')

    # find all files in the folder that end with .dot
    files = glob.glob(os.path.join(folder_path, '*.dot'))

    # iterate over the list of filepaths & remove each file.
    for file in files:
        try:
            os.remove(file)
            print(f'{file} has been removed successfully')
        except Exception as e:
            print(f'Error occurred while deleting {file}. Error message: {str(e)}')