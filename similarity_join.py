import re
import pandas as pd

class SimilarityJoin:
    def __init__(self, data_file1, data_file2):
        self.df1 = pd.read_csv(data_file1)
        self.df2 = pd.read_csv(data_file2)

    def formats(self, rows):
        #split and lower-case
        words = re.split(r'\W+', rows)
        return [word.lower() for word in words if word.lower() != '']

    def preprocess_df(self, df, cols): 
        #concat cols
        df['joinKey'] = df[cols[0]].astype(str).str.cat(df[cols[1]].astype(str),sep=' ')
        #drop None/nan 
        df['joinKey'] = df['joinKey'].str.replace(' nan', '').dropna()
        df['joinKey'] = df['joinKey'].apply(self.formats)
        return df
        
    def filtering(self, df1, df2):
        #expand joinkeys
        df1['expand'] = df1['joinKey']
        ex1 = df1.explode('expand')
        df2['expand'] = df2['joinKey']
        ex2 = df2.explode('expand')
        
        #merge shared items
        merged = pd.merge(ex1,ex2, on='expand', how='inner')
        rename = merged.rename(columns={'id_x': 'id1', 'id_y': 'id2', 'joinKey_x': 'joinKey1','joinKey_y': 'joinKey2'})
        
        #filter duplicates
        filtered = rename.drop_duplicates(['id1', 'id2'], keep='first')
        cand_df = filtered[['id1', 'id2','joinKey1', 'joinKey2']]
        return cand_df
    
    def jaccard(self, cols):
        #get jaccard value
        jk1 = set(cols[0])
        jk2 = set(cols[1])
        intersects = len(jk1 & jk2)
        unions = len(jk1 | jk2)
        result = 1.0 * intersects / unions
        return result

    def verification(self, cand_df, threshold):
        cand_df['jaccard'] = cand_df[['joinKey1', 'joinKey2']].apply(self.jaccard, axis=1)        
        #pick cells with threshold
        result_df = cand_df[cand_df['jaccard'] >= threshold]
        return result_df

    def evaluate(self, result, ground_truth):
        result_set = set([(x[0]+x[1]) for x in result])
        ground_set = set([(x[0]+x[1]) for x in ground_truth])
        #true match
        T = len(result_set & ground_set)

        precision = 1.0 * T / len(result)
        recall = 1.0 * T / len(ground_truth)
        fmeasure = 2 * precision * recall / (precision+recall)
        return (precision, recall, fmeasure)

    def jaccard_join(self, cols1, cols2, threshold):
        new_df1 = self.preprocess_df(self.df1, cols1)
        new_df2 = self.preprocess_df(self.df2, cols2)
        print ("Before filtering: %d pairs in total" %(self.df1.shape[0] *self.df2.shape[0])) 

        cand_df = self.filtering(new_df1, new_df2)
        print ("After Filtering: %d pairs left" %(cand_df.shape[0]))

        result_df = self.verification(cand_df, threshold)
        print ("After Verification: %d similar pairs" %(result_df.shape[0]))

        return result_df

if __name__ == "__main__":
    er = SimilarityJoin("Amazon_sample.csv", "Google_sample.csv")
    amazon_cols = ["title", "manufacturer"]
    google_cols = ["name", "manufacturer"]
    result_df = er.jaccard_join(amazon_cols, google_cols, 0.5)

    result = result_df[['id1', 'id2']].values.tolist()
    ground_truth = pd.read_csv("Amazon_Google_perfectMapping_sample.csv").values.tolist()
    print ("(precision, recall, fmeasure) = ", er.evaluate(result, ground_truth))