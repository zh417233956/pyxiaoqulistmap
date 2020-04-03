from flask import request, Response, json
from routes import app
from utils.clientresult import clientresult
import traceback
from handlers.searchHanler import Search
import logging


@app.route('/api/search')
def index():
    try:
        q = request.args.get("q")
        _result = Search().search(q)
        result = clientresult.success(_result, "检索成功")
        pass
    except Exception as e:
        # msg = traceback.format_exc()
        result = clientresult.error("请求异常:" + repr(e))
        pass
    content = json.dumps(result.__dict__)
    resp = Response_headers(content)
    return resp


def Response_headers(content):
    '''
    定义返回数据头
    '''
    resp = Response(content)
    """
    所有域名都可以调用
    """
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
