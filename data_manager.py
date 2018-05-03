import pickle
from sklearn.externals import joblib

class DataManager():
    # train된 tfidf 오브젝트
    tfidf_obj = None

    # corenet definition data
    corenet_data = None

    # corenet object data
    corenet_obj = None

    # korterm tfidf object
    korenet_tfidf = None

    isInitialized = False

    @staticmethod
    def init_data():
        DataManager.tfidf_obj = joblib.load('./data/trained_tfidf_etri_tokenize.pkl')
        DataManager.corenet_obj = pickle.load(open('./data/corenet_lemma_info_obj_with_freq.pickle', 'rb'))
        DataManager.korenet_tfidf = joblib.load('./data/korterm_tfidf.pickle')
        DataManager.korterm_cooccur_freq = pickle.load(open('./data/korterm_cooccur_freq_014_final.pickle', 'rb'))
        DataManager.korterm_shortest_path = pickle.load(open('./data/korterm_shortest_path.pickle', 'rb'))
        DataManager.isInitialized = True
