import time
import disambiguater
import corenet
from data_manager import DataManager
from enum import Enum
from mrf_word_sense_disambiguation import MRFWordSenseDisambiguation

class WSDMode(Enum):
    TF_IDF = 1
    MRF = 2

class WSD:
    @staticmethod
    def init_data():
        DataManager.init_data()
        # wordnet 최초 호출이 오래 걸려서 미리 한 번 불러서 초기화 해 놓음
        corenet.getWordnet("쓰다", 3.0, 1.0, only_synonym=True)

    def _extract_disambiguate_obj_from_text(self,input_text):
        beginIdx = 0
        endIdx = 0
        word = ""
        count = 0
        bracket_opend = False
        for char in input_text:
            if char == "[":
                beginIdx = count
                bracket_opend = True
            elif char == "]":
                endIdx = count
                break
            else:
                count += 1
                if (bracket_opend):
                    word = word + char
        text = input_text.replace('[', '').replace(']', '')
        return {
            'text': text,
            'word': word,
            'beginIdx': beginIdx,
            'endIdx': endIdx
        }

    def disambiguate(self,sent,mode=WSDMode.TF_IDF):
        input = self._extract_disambiguate_obj_from_text(sent)

        korterm, origin_word, score = 'Error', '', 0.0
        try:
            if mode == WSDMode.TF_IDF:
                disambig = disambiguater.TFIDFDisambiguater()
                korterm, origin_word, cluster_num, score = disambig.disambiguate(input)
            else:
                mrfdis = MRFWordSenseDisambiguation()
                korterm, origin_word, cluster_num, score = mrfdis.disambiguate(input)

            def_cluster = DataManager.corenet_obj[origin_word][cluster_num]
            korterm_list = list(def_cluster['korterm_set'])
            def_usuage_list = list(def_cluster['definition_set'])
            def_usuage_list.extend(list(def_cluster['usuage_set']))
            debug = 1
        except:
            korterm_list = []
            def_usuage_list = []

        return {
            'korterm' : korterm,
            'origin_word' : origin_word,
            'korterm_list' : korterm_list,
            'def_usuage_list' : def_usuage_list,
            'score' : score
        }


if __name__ == '__main__':
    st_time = int(time.time())
    WSD.init_data()
    print ('data loaded : %d sec'%(int(time.time())-st_time))
    wsd = WSD()
    result1 = wsd.disambiguate("그는 모자와 안경을 [쓰고] 있었다.",  mode=WSDMode.TF_IDF)
    result2 = wsd.disambiguate("연필로 글씨를 [썼다].", mode=WSDMode.TF_IDF)
    print (result1)
    print (result2)
    DEBUG = 1



