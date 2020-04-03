class clientresult(object):
    def __init__(self, status: str, result: object, msg: str):
        self.status = status
        self.result = result
        self.msg = msg

        pass

    @staticmethod
    def success(result: object, msg=""):
        # _res = object
        # 类型判断 list、tuple、dict、set
        # if isinstance(result, list):
        #     # _res = result.__dict__
        #     pass
        return clientresult("success", result, msg)

    @staticmethod
    def error(errormsg: str):
        return clientresult("fail", "", errormsg)
