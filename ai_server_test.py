## 선행 내용
# flask json, pymysql 등 설치
#
# AWS 서비스와 상호작용하기 위해 필요한 절차
# 'pip install boto3'   설치
# 'pip3 install awscli' 설치
# 'aws configure'       계정 등록 하기 위한 명령어
#    AWS AccessKey ID :         AKIA6CYUKORF5PGRSLWA
#    AWS Secret Access Key :    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
#    Default region name :      ap-northeast-2
#    Default output format :    None
# 다 입력하면 AWS 계정 등록 완료됨. python에서 AWS 접근 가능


from flask import Flask, request
import json
import boto3
import pymysql
from main_ocr import main_ocr


# SQL 쿼리 결과 리턴
def send_sql(query):
    ai_db = pymysql.connect(
        user='admin',
        passwd='12345678',
        host='hansung-ai-database.cl49gzqg3xoc.ap-northeast-2.rds.amazonaws.com',
        db='ai_db',
        charset='utf8'
    )
    cursor = ai_db.cursor(pymysql.cursors.DictCursor)
    cursor.execute(query)
    result = cursor.fetchall()
    ai_db.commit()      # INSERT 등은 commit 해줘야 반영됨

    return result   # list 형식으로 반환됨


# 데이터 포맷
def format_result(_list, _mode = 0):        # mode 0:accounts 용, 1:calendars 용
    length = len(_list)      # 등록 건수
    total = 0                # 합계 금액
    for item in _list:
        if(_mode==0):
            total += item['price']
        elif(_mode==1):
            total += item['total']

    json = {
        'total' : total,    # 합계 금액
        'length' : length,  # 등록 건수
        'list' : _list      # 데이터 목록
        }

    return json


# 버켓에서 가져온 파일 로컬에 저장
def get_bucket_file(file_name):
    # s3.download_file('버켓이름','버켓하위 경로를 포함한 s3속 파일이름',"로컬에 저장할때 파일이름")
    s3 = boto3.client('s3')
    s3.download_file('hansung-ai-bucket', file_name, "target.jpg")

    return "target.jpg"     # 로컬에 저장된 파일명 리턴

    

app = Flask(__name__)


##### 전체 지출 목록
@app.route('/')
@app.route('/accounts')
def api_accounts():
    query = 'SELECT * FROM account_table order by date'          
    result = send_sql(query)

    json = format_result(result)
    json['result'] = 'SUCCESS'

    return json


# url에 입력된 년-월의 데이터 목록 가져오기
@app.route('/accounts/<yearmonth>')
def api_accounts_yearmonth(yearmonth):
    query = ''' SELECT * FROM account_table WHERE DATE_FORMAT(date,'%Y-%m') = '{}' order by date'''.format(yearmonth)                      
    result = send_sql(query)
    
    json = format_result(result)
    json['result'] = 'SUCCESS'

    return json


# accounts 목록 데이터 삽입
@app.route('/accounts', methods=['POST'])
def api_insert():
    #print(request.is_json)
    params = request.get_json()
    #print(params)
    
    date = params['date']
    place = params['place']
    price = params['price']

    query = ''' INSERT INTO account_table(date, place, price)
                VALUES ('{}', '{}', {}) '''.format(date, place, price)
    result = send_sql(query)

    return {'result':'SUCCESS'}


# accounts 목록 데이터 삭제
@app.route('/accounts/<idx>', methods=['DELETE'])
def api_delete(idx):
    print('DELETE 호출! idx : '+ idx)
    query = 'DELETE FROM account_table WHERE idx = {}'.format(idx)
    result = send_sql(query)

    return {'result':'SUCCESS'}


# 달력용
@app.route('/calendars')
def api_calendars():
    query = 'SELECT * FROM calendar_view'          
    result = send_sql(query)
    print(result)

    json = format_result(result, 1)
    json['result'] = 'SUCCESS'

    return json


@app.route('/calendars/<yearmonth>')
def api_calendars_yearmonth(yearmonth):
    query = ''' SELECT * FROM calendar_view WHERE DATE_FORMAT(date,'%Y-%m') = '{}' '''.format(yearmonth)          
    result = send_sql(query)
    print(result)

    json = format_result(result, 1)
    json['result'] = 'SUCCESS'

    return json



# GET URI 내에 이미지 파일명 파라미터로 넘겨받음
@app.route('/recognition/<img>')
def api_recognition(img):
    params = request.get_json()

    # 버킷에서 이미지 가져와서 새로운 이름으로 저장한 파일명
    # 해당 이미지를 가지고 텍스트 인식 필요
    img = get_bucket_file(img)
        
    date = '2022-05-05'
    place = 'place'
    price = "0"

    price, date = main_ocr(img)

    # price, date 는 String 타입
    # main_ocr 함수는 main_ocr.py 에 정의되어있음
    # price, date 찾는 함수는 find_string.py 에 정의되어있음


    #
    #
    #
    #   여기에 img를 가지고 텍스트 인식 후,
    #   추출된 날짜, 장소(매장), 금액을 date, place, price 에 저장하여 josn 형식으로 리턴해야 함
    #   리턴 형식은 아래 return과 같으므로 위 변수에 값을 저장만 하면 됨
    #
    #
    #
    
    return {'result':'SUCCESS', 'date':date, 'place':place, 'price':price}


@app.errorhandler(404) 
def error_handling_404(error): 
    return {'result':'FAIL', 'code':404}


@app.errorhandler(500)
def error_handling_500(error):
    return {'result':'FAIL', 'code':500}



if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False     # 한글 깨짐 처리
    app.run(host='0.0.0.0', port=8000)
