import bm25s
import json, re
import datasets
import Stemmer 

def load_corpus(corpus_path: str):
    """Load corpus using datasets library"""
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus


def load_docs(corpus, doc_idxs):
    """Load documents by indices"""
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

class BM25RetrieverLunce:
    def __init__(self, corpus_path: str):
        
        self.retriever = self._build_index(corpus_path)
        self.corpus = load_corpus(corpus_path=corpus_path)
    
    def _build_index(self, corpus_path):
        with open(corpus_path,"r") as file:
            lines = file.readlines()
        self.raw_data = []
        for line in lines:
            try:
                data = json.loads(line)
                self.raw_data.append(data)
            except:
                print(f"error when loading: {data}")
        corpus = [re.sub(r'[^\w\s]', '', data["contents"]) for data  in self.raw_data] 
        self.stemmer = Stemmer.Stemmer("english")
        retriever =  bm25s.BM25() #corpus=corpus
        retriever.index(bm25s.tokenize(corpus, stopwords="en", stemmer=self.stemmer))
        return retriever
    
    def _search(self,query: str, num: int):
        results, scores = self.retriever.retrieve(bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer), k=num)
        return results[0], scores[0]
    
    
    
if __name__ == "__main__":
    bm25_ = BM25Retriever("/mnt/nas/alex/deep-research/src/rag_setup/data/corpus/corpus.jsonl")
    print(bm25_._search("Mc Donald", 5))
    result = bm25_._search(" Donald", 5)
    print(load_docs(bm25_.corpus, result[0][0]))