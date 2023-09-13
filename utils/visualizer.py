import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Visualizer:
    def __init__(self) -> None:
        pass
    def visualize(self) -> None:
        pass

class BucketVisualizer(Visualizer):
    def __init__(self,bucket_col,num_buckets,inner_buckets) -> None:
        super().__init__()
        self.bucket_col = bucket_col
        self.num_buckets = num_buckets
        self.inner_buckets = inner_buckets
    def visualize(self,df,labeler) -> None:
        
        # Create the bins
        bins = [df[self.bucket_col].quantile(i/self.num_buckets) for i in range(self.num_buckets+1)]
        
        # Create the labels
        labels = [f'Bucket_{i}' for i in range(1, self.num_buckets+1)]
        
        # Create a new feature that represents the bucket each record falls into
        df['Bucket'] = pd.cut(df[self.bucket_col], bins=bins, labels=labels)
        
        # Plot each bucket
        for i in range(1, self.num_buckets+1):
            bucket_df = df[df['Bucket'] == f'Bucket_{i}']
            # bucket_df.drop(["low"],axis=1).plot(alpha=0.5, figsize=(10, 7))
            bucket_df.loc[:,f'{self.inner_buckets[0]}_{self.num_buckets}bucket'] = pd.qcut(bucket_df[self.inner_buckets[0]], q=self.num_buckets, labels=[f'Bucket_{i}' for i in range(1, self.num_buckets+1)])
            bucket_df.loc[:,f'{self.inner_buckets[1]}_{self.num_buckets}bucket'] = pd.qcut(bucket_df[self.inner_buckets[1]], q=self.num_buckets, labels=[f'Bucket_{i}' for i in range(1, self.num_buckets+1)])
            bucket_df = bucket_df.merge(labeler.actual_returns,left_index=True,right_index=True)
            print(bucket_df)
            pivot_df = pd.pivot_table(bucket_df,index=f'{self.inner_buckets[0]}_{self.num_buckets}bucket', columns=f'{self.inner_buckets[1]}_{self.num_buckets}bucket', values="return",aggfunc=lambda x: x.mean()/x.std())
            print(pivot_df)
            sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0.0)
            plt.title(f'Bucket_{i}_of_{self.bucket_col}')
            plt.show()

class TwoDimBucketVisualizer(Visualizer):
    def __init__(self,bucket_col,num_buckets) -> None:
        super().__init__()
        self.bucket_col = bucket_col
        self.num_buckets = num_buckets
    def visualize(self,df,labeler) -> None:
        print(df)
        df.loc[:,f'{self.bucket_col[0]}_{self.num_buckets}bucket'] = pd.qcut(df[self.bucket_col[0]], q=self.num_buckets, labels=[f'Bucket_{i}' for i in range(1, self.num_buckets+1)])
        df.loc[:,f'{self.bucket_col[1]}_{self.num_buckets}bucket'] = pd.qcut(df[self.bucket_col[1]], q=self.num_buckets, labels=[f'Bucket_{i}' for i in range(1, self.num_buckets+1)])
        df = df.merge(labeler.actual_returns,left_index=True,right_index=True)
        pivot_df = pd.pivot_table(df,index=f'{self.bucket_col[0]}_{self.num_buckets}bucket', columns=f'{self.bucket_col[1]}_{self.num_buckets}bucket', values="return",aggfunc=lambda x: x.mean()/x.std())
        print(pivot_df)
        sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', center=0.0)
        plt.title(f'2d Heatmap of {self.bucket_col}')
        plt.show()