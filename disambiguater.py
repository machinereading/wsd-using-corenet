import data_util
from data_manager import DataManager
from sklearn.metrics.pairwise import cosine_similarity

class Disambiguater:
    '''
    문장과 문장 내의 특정 단어가 주어졌을 때, 
    이 단어의 문맥 상에서의 의미와 일치하는 CoreNet 상의 표제어 및 어깨번호(의미번호) 를 반환하는 기능을 수행하는 모듈들의
    추상 클래스. 각 Disambiguater 모듈들은 이 모듈을 상속받아서 정의된 interface를 구현한다.
    '''
    def disambiguate(self, input):
        '''
        주어젠 입력에 알맞은 표제어 및 어깨번호를 반환한다.
        Input : Dictionary
        -- text : 주어진 문장
        -- word : 주어진 문장에서 CoreNet 상에 알맞은 정보를 찾고자 하는 어휘
        -- beginIdx : 해당 word가 주어진 문장에서 시작하는 위치
        -- endIdx : 해당 word가 주어진 문장에서 끝나는 위치
        Output : Array of dictionary
        -- lemma : 표제어
        -- sensid : 어깨번호
        -- definition : 정의 
        '''
        return []

    def get_def_candidate_list(self, word):
        '''
        주어진 word의 후보가 될 수 있는 definition list를 가져온다.
        기본적으로 정확히 일치하는 것만 가져온다.
        '''
        return data_util.get_corenet_matching_def_list(word)

    def get_wsd_word_list(self, input):
        inf = 99999
        nlp_result = data_util.get_nlp_test_result(input['sent'])
        if (nlp_result == None):
            return []

        idx_mapping = {0:0}
        word_count = 0
        byteIdx = 0
        for character in input['sent']:
            word_count += 1
            byteIdx += data_util.get_text_length_in_byte(character)
            idx_mapping[byteIdx] = word_count
        idx_mapping[inf] = word_count

        wsd_word_list = []
        morphs = nlp_result['sentence'][0]['WSD']
        morphs.append({'position': inf})
        for i in range(len(morphs) - 1):
            morp = morphs[i]
            # 동사 or 형용사 일 경우 wordnet 포맷에 맞추어 원형+'다' 형태로 반환한다. e.g.) '멋있' + '다'
            if (morp['type'] == 'VA' or morp['type'] == 'VV' or morp['type'] == 'NNG'):
                st = idx_mapping[ morp['position'] ]
                en = st + len(morp['text']) - 1
                if (morp['type'] == 'VA' or morp['type'] == 'VV'):
                    morp['text'] = morp['text'] + '다'
                wsd_word_list.append({'word':morp['text'], 'st':st, 'en':en})

        return wsd_word_list


class TFIDFDisambiguater(Disambiguater):
    '''
    Kortermnum Disambiguater
    '''
    def disambiguate(self, input):
        if not DataManager.isInitialized:
            return []

        sent = input['sent']
        input_vector = DataManager.tfidf_obj.transform([sent])

        wsd_word_list = self.get_wsd_word_list(input)

        result = []
        for word_cand in wsd_word_list:
            word = word_cand['word']
            korterm_list = []
            if (word not in DataManager.corenet_obj):
                continue

            cand_list = DataManager.corenet_obj[word]
            for idx in range(len(cand_list)):
                item = cand_list[idx]
                list = item['korterm_set']
                for korterm in list:
                    korterm_list.append({'korterm': korterm, 'idx': idx})

            max_cos_similiarity = -10000.0
            max_korterm = ''
            max_idx = '0'

            for korterm_item in korterm_list:
                korterm = korterm_item['korterm']
                index = korterm_item['idx']
                if (korterm not in DataManager.korenet_tfidf):
                    continue
                if (max_korterm == korterm):
                    continue

                korterm_vec = DataManager.korenet_tfidf[korterm]
                cos_similarity = cosine_similarity(input_vector, korterm_vec)[0][0]

                if (cos_similarity > max_cos_similiarity):
                    max_cos_similiarity, max_korterm, max_idx = cos_similarity, korterm, index
            result.append({'word':word, 'st':word_cand['st'], 'en':word_cand['en'], 'sense_num':index, 'conf':max_cos_similiarity})
        return result

if __name__ == "__main__":
    DataManager.init_data()
    d = TFIDFDisambiguater().disambiguate({'sent':'그가 대상을 받았다.'})
    print (d)
    debug = 1
