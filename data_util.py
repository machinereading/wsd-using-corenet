# -*- coding: utf-8 -*-
import corenet
import time
import json
import urllib.request

tokenize_count = 0
token_start_time = 0
hannanumTagger = None

def get_nlp_test_result(text):
    '''
    주어진 텍스트에 대해서 ETRI 텍스트 분석 결과를 반환한다. 
    '''
    etri_pos_url = 'http://143.248.135.20:22334/controller/service/etri_parser '
    data = "sentence="+text
    try:
        req = urllib.request.Request(etri_pos_url, data=data.encode('utf-8'))
        response = urllib.request.urlopen(req)
        result = response.read().decode('utf-8')
        result = json.loads(result)
    except:
        #print('error text : ' + text)
        return None
    return result

def get_pos_tag_result(text):
    '''
    주어진 텍스트에 대해서 ETRI 텍스트 분석 결과를 반환한다. 
    '''
    etri_pos_url = 'http://143.248.135.60:31235/etri_pos'
    data = '{"text":"' + text + '"}'
    try:
        req = urllib.request.Request(etri_pos_url, data=data.encode('utf-8'))
        response = urllib.request.urlopen(req)
        result = response.read().decode('utf-8')
        result = json.loads(result)
    except:
        print ('error text : ' + text)
        return []
    return result

def etri_tokenizer(text):
    '''
    ETRI 형태소 분석 결과를 기반으로 text를 Tokenize한다.
    '''
    global tokenize_count, token_start_time
    word_list = []

    pos_tag_result = get_nlp_test_result(text)
    if (pos_tag_result is None):
        return []
    pos_tag_result = pos_tag_result['sentence']
    if (pos_tag_result is None):
        return []

    for sent in pos_tag_result:
        morph_list = sent['morp']
        for morph in morph_list:
            if (morph['type'][0] == 'S'): # 부호는 무시
                continue
            word_list.append(morph['lemma'])

    tokenize_count += 1
    if ((tokenize_count % 1000) == 0 ):
        print('%d tokenize finished %.2f second elpased'%(tokenize_count, time.time()-token_start_time))
        token_start_time = time.time()

    return word_list

def read_corenet_definition_data():
    '''
      idx : 일련번호
      term : 표제어
      vocnum : 동음 형용사에 대한 표제어 구분 순번
      semnum : 의미번호
      definition1 : 의미풀이
      definition2 : 풀이된말
      usage : 예문
    '''
    f = open('./data/definition.dat', 'r', encoding='utf-8')

    definition_list = []
    i = 0
    sttime = time.time()
    for line in f:
        items = line.strip().split('\t')
        definition_list.append({
            'id': int(items[0]),
            'term': items[1],
            'vocnum': int(items[2]),
            'semnum': int(items[3]),
            'definition1': items[4] if len(items) > 4 else '',
            'definition2': items[5] if len(items) > 5 else '',
            'usuage': items[5] if len(items) > 6 else ''
        })

        i += 1
        if (i % 100000 == 0):
            print(str(i) + 'corenet data loaded...')

    f.close()
    return definition_list

def convert_deflist_to_sent_list(definition_list):
    '''
    입력으로 받은 definition 데이터 리스트를 단어들의 집합으로 이루어진 문장 목록으로 변환
    '''
    sent_list = []
    for definiton in definition_list:
        text = convert_def_to_sentence(definiton)
        if (len(text) > 0):
            sent_list.append(text)
    return sent_list


def convert_def_to_sentence(definiton):
    '''
    core net defintion에 있는 문장들을 합쳐서 하나의 문장으로 만든다.
    '''
    term = definiton['term']
    text = term + '. ' \
           + definiton['definition1'].replace('～', term) + ' ' \
           + definiton['definition2'].replace('～', term) + ' ' \
           + definiton['usuage'].replace('～', term)
    return text


def get_text_length_in_byte(text):
    return len(str.encode(text))


def get_real_corenet_matching_def_list(word):
    matching_def_list = []
    try:
        matching_corenet_list = corenet.getRealCoreNet(word)
    except:
        return []

    for corenet_item in matching_corenet_list:
        #Nan check. vocnum이나 semnum이 float 중 nan이면 스킵한다
        if (corenet_item['semnum'] != corenet_item['semnum'] or
            corenet_item['vocnum'] != corenet_item['vocnum']):
            continue
        definition1, usuage = corenet.getDefinitionAndUsuage(word, corenet_item['vocnum'], corenet_item['semnum'])

        if (type(definition1) == float):
            definition1 = ''
        if (type(usuage) == float):
            usuage = ''
        if (type(corenet_item['kortermnum']) == float):
            corenet_item['kortermnum'] = ''

        item = {
            'term' : word,
            'vocnum' : int(corenet_item['vocnum']),
            'semnum' : int(corenet_item['semnum']),
            'definition1' : definition1,
            'definition2' : '',
            'kortermnum' : corenet_item['kortermnum'],
            'usuage' : usuage,
        }
        matching_def_list.append(item)
    return matching_def_list

def get_corenet_matching_def_list(word):
    '''
    주어진 word와 일치하는 corenet 상의 표제어들의 정의 목록을 반환한다. 
    '''
    matching_def_list = []

    try :
        matching_semnum_list = corenet.getSemnum(word)
    except:
        return []

    for semnum in matching_semnum_list:
        definition1 = corenet.getDefinition(word, semnum['vocnum'], semnum['semnum'])
        if (type(definition1) == float):
            definition1 = ''
        if (definition1 == 'None'):
            definition1 = ''

        usuage = corenet.getUsage(word, semnum['vocnum'], semnum['semnum'])
        if (type(usuage) == float):
            usuage = ''
        if (usuage == 'None'):
            usuage = ''
        pos = corenet.getPos(word, semnum['vocnum'], semnum['semnum'])

        item = {
            'term' : word,
            'vocnum' : int(semnum['vocnum']),
            'semnum' : int(semnum['semnum']),
            'definition1' : definition1,
            'definition2' : '',
            'usuage' : usuage,
            'pos' : pos
        }
        matching_def_list.append(item)

    return matching_def_list


def get_hanwoo_dic_matching_def_list(word):
    '''
    주어진 word와 일치하는 corenet 상의 표제어들의 정의 목록을 반환한다. 
    '''
    matching_def_list = []

    try :
        matching_corenet_list = corenet.getCoreNet(word)
    except:
        return []
    for corenet_data in matching_corenet_list:
        definition1 = corenet_data['definition']
        if (type(definition1) == float or definition1 == 'None'):
            definition1 = ''
        usuage = corenet_data['usage']
        if (type(usuage) == float or usuage == 'None'):
           usuage = ''

        item = {
            'term': word,
            'vocnum': int(corenet_data['vocnum']),
            'semnum': int(corenet_data['semnum']),
            'definition1': definition1,
            'definition2': '',
            'usuage': usuage,
            'pos': corenet_data['pos']
        }
        matching_def_list.append(item)

    return matching_def_list



if __name__ == '__main__':
    a = 1
