""" 利用倒排表进行优化 """
import pandas as pd
import matplotlib as mpl
import numpy as np
from nltk.probability import FreqDist
import time
from utils.jiebaSegment import Seg
from utils.sentenceSimilarity import SentenceSimilarity
import logging

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # enable chinese


class Search():
    def read_corpus(self, seg):
        qList = []
        # 问题的关键词列表
        qList_kw = []
        aList = []
        try:
            data = pd.read_csv('./data/芒果小区列表.txt', header=None)
            data_ls = np.array(data).tolist()
            for t in data_ls:
                qList.append(t[0])
                qList_kw.append(seg.cut(str(t[0])))
                aList.append(t[1])
            pass
        except Exception as ex:
            logging.error(ex)
            pass
        return qList_kw, qList, aList

    def plot_words(self, wordList):
        fDist = FreqDist(wordList)
        # logging.info("单词总数: ", fDist.N())
        # logging.info("不同单词数: ", fDist.B())
        fDist.plot(10)

    def invert_idxTable(self, qList_kw):  # 定一个一个简单的倒排表
        invertTable = {}
        for idx, tmpLst in enumerate(qList_kw):
            for kw in tmpLst:
                if kw in invertTable.keys():
                    invertTable[kw].append(idx)
                else:
                    invertTable[kw] = [idx]
        return invertTable

    def filter_questionByInvertTab(self, inputQuestionKW, questionList,
                                   answerList, invertTable):
        idxLst = []
        questions = []
        answers = []
        for kw in inputQuestionKW:
            if kw in invertTable.keys():
                idxLst.extend(invertTable[kw])
        idxSet = set(idxLst)
        for idx in idxSet:
            questions.append(questionList[idx])
            answers.append(answerList[idx])
        return questions, answers

    def search(self, question):
        result = {}
        time1 = time.time()
        # 设置外部词
        seg = Seg()
        seg.load_userdict('./configs/userdict.txt')
        # 读取数据
        qList_kw, questionList, answerList = self.read_corpus(seg)
        """简单的倒排索引"""
        # 计算倒排表
        invertTable = self.invert_idxTable(qList_kw)
        inputQuestionKW = seg.cut(question)
        # 利用关键词匹配得到与原来相似的集合
        questionList_s, answerList_s = self.filter_questionByInvertTab(
            inputQuestionKW, questionList, answerList, invertTable)
        if len(questionList_s) > 1:
            # 初始化模型
            try:
                ss = SentenceSimilarity(seg)
                ss.set_sentences(questionList_s)
                ss.TfidfModel()  # tfidf模型
                # ss.LsiModel()         # lsi模型
                # ss.LdaModel()         # lda模型

                question_k = ss.similarity_k(question, 5)
                result["anwser"] = []
                for idx, score in zip(*question_k):
                    result["anwser"].append({
                        "title": questionList_s[idx],
                        "id": answerList_s[idx],
                        "score": str(score)
                    })
                pass
            except Exception as ex:
                logging.error(f'分析失败{ex}')
                result["anwser"] = {
                    "title": questionList_s[0],
                    "id": answerList_s[0],
                    "score": "0"
                }
                pass
        elif len(questionList_s) == 1:
            result["anwser"] = {
                "title": questionList_s[0],
                "id": answerList_s[0],
                "score": "0"
            }
        else:
            result["anwser"] = []
            pass
        pass
        time2 = time.time()
        cost = time2 - time1
        result["cost"] = cost
        return result

    def xiaoqu_list_map(self, topstr, first):
        result = []
        top = int(topstr)
        time1 = time.time()
        # 设置外部词
        seg = Seg()
        seg.load_userdict('./configs/userdict.txt')
        # 读取数据
        qList_kw, questionList, answerList = self.read_corpus(seg)
        # 简单的倒排索引
        # 计算倒排表
        invertTable = self.invert_idxTable(qList_kw)
        search_list = self.read_xiaoqu_list(seg, 1)
        search_index = 0
        try:
            for itemq in search_list:
                if top != 0 and search_index > top:
                    break
                search_index += 1
                items = self.get_list_map(seg, itemq, invertTable, qList_kw,
                                          questionList, answerList)
                if first == "true":
                    if len(items["anwser"]) > 0:
                        result.append({
                            "q": itemq,
                            "s": items["anwser"][0]["id"],
                            "t": items["anwser"][0]["title"],
                            "score": items["anwser"][0]["score"]
                        })
                        pass
                    else:
                        result.append({"q": itemq, "s": {}})
                    pass
                else:
                    result.append({"q": itemq, "s": items})
            pass
        except Exception as ex:
            logging.error(ex)
            pass

        time2 = time.time()
        cost = time2 - time1
        logging.error(f"耗时：{cost}")
        return result

    def get_list_map(self, seg, question, invertTable, qList_kw, questionList,
                     answerList):
        result = {}
        # 开始匹配
        inputQuestionKW = seg.cut(question)
        # 利用关键词匹配得到与原来相似的集合
        questionList_s, answerList_s = self.filter_questionByInvertTab(
            inputQuestionKW, questionList, answerList, invertTable)
        result["anwser"] = []
        if len(questionList_s) > 1:
            # 初始化模型
            try:
                ss = SentenceSimilarity(seg)
                ss.set_sentences(questionList_s)
                ss.TfidfModel()  # tfidf模型
                # ss.LsiModel()         # lsi模型
                # ss.LdaModel()         # lda模型

                question_k = ss.similarity_k(question, 5)
                for idx, score in zip(*question_k):
                    result["anwser"].append({
                        "title": questionList_s[idx],
                        "id": answerList_s[idx],
                        "score": str(score)
                    })
                pass
            except Exception as ex:
                logging.error(f'分析失败{ex}')
                result["anwser"].append({
                    "title": questionList_s[0],
                    "id": answerList_s[0],
                    "score": "0"
                })
                pass
        elif len(questionList_s) == 1:
            result["anwser"].append({
                "title": questionList_s[0],
                "id": answerList_s[0],
                "score": "0"
            })
        else:
            result["anwser"] = []
            pass
        pass
        return result

    def read_xiaoqu_list(self, seg, types):
        xqList = []

        xq_path = "./data/芒果小区列表.txt"
        if types == 1:
            xq_path = "./data/贝壳小区列表.txt"

        data = pd.read_csv(xq_path, header=None)
        data_ls = np.array(data).tolist()
        for t in data_ls:
            xqList.append(t[0])
        return xqList
