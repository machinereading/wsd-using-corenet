from wsd import WSD, WSDMode

if __name__ == '__main__':
    # WSD 클래스를 사용하기 전에 데이터를 메모리에 불러오기 위해
    # 한 번 꼭 호출해 주어야 합니다.
    # 이 부분이 5~8초 정도 걸립니다.
    WSD.init_data()

    # 개체 생성
    wsd = WSD()

    # TF-IDF 모드로 호출하는 예시
    result1 = wsd.disambiguate("그는 모자와 안경을 [쓰고] 있었다.",  mode=WSDMode.TF_IDF)
    result2 = wsd.disambiguate("연필로 글씨를 [썼다].", mode=WSDMode.TF_IDF)
    print(result1)
    print(result2)

    # MRF 모드로 호출하는 예시
    result3 = wsd.disambiguate("자고산 서쪽 [사면]은 낙동강과 닿아있고 그는 죄의 사면을 요구했다.", mode=WSDMode.MRF)
    result4 = wsd.disambiguate("자고산 서쪽 사면은 낙동강과 닿아있고 그는 죄의 [사면]을 요구했다.", mode=WSDMode.MRF)
    print(result3)
    print(result4)


