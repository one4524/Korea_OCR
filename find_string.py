import re
import datetime


def find_sum_n_date(text_boxs, text):
    date = ""
    str = ""
    for i, text_line in enumerate(text_boxs):
        line = []

        for num, t in enumerate(text_line):
            line.append(text[num])
            str += text[num]

        str = string_to_number(str)

        print(i + 1, "줄 텍스트 : ", str)

        price_bool, str_sum = is_price(str)

        if price_bool:
            price = "".join(str_sum)
            print("합계 : ", price)
        else:
            print("합계 없음")

        date_bool, date = is_data(str)

        if date_bool:
            print("날짜 : ", date)
        else:
            print("날짜 없음")

        line.clear()
        str = ""


# 영문자o,O -> 숫자 0 으로 변환
# 필요없는 특수기호, 문자 등 삭제하고 숫자만 추출 : 정규표현식
def string_to_number(str):
    str = str.replace('o', '0')  # 소문자'o' 숫자'0' 변환
    str = str.replace('O', '0')  # 대문자'O' 숫자'0' 변환
    str = str.replace('ㅇ', '0')  # 한글'ㅇ' 숫자'0' 변환

    return str


# 금액과 관련된 문자열이 맞는지 판단
# 합계, 결제, 금액, 등의 단어가 들어 있으면 true 리턴
def is_price(str):
    sum_text = ['금액', '결제', '합계', '함계', '함게', '합게', '압계', '압게', '암계', '암게', '힙켸', '입켸', '입케', '힙케']
    bool = False
    str_num = []

    for sum in sum_text:
        if str.find(sum) != -1:
            str_num = re.findall("\d+", str)
            bool = True

    return bool, str_num


def is_data(str):
    date_text = ['판매', '거래', '승인', '일자', '발행']
    bool = False

    str = re.sub('[년연넌언월원운언/.,-]', "-", str)  # 하이픈으로 변환

    date_type= [r'\d{4}-\d{1,2}-\d{2}', r'\d{2}-\d{1,2}-\d{2}', r'\d{4}\d{1,2}-\d{2}', r'\d{4}-\d{1,2}\d{2}',
                r'\d{4}\d{2}\d{2}']

    for dt in date_type:

        list = re.findall(dt, str)  # 0000/00/00 년도가 4개인 것 확인( : 고려x)

        if len(list) != 0:
            bool = True
            break

    result = '20' + list[0] if len(list[0]) == 8 else list[0]  # 00-00-00 -> 2000-00-00

    return bool, datetime.datetime.strptime(result, '%Y-%m-%d')  # 리턴 형태 : datetime
