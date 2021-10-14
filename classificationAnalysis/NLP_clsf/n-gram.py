# -*- coding: UTF-8 -*-

"""
  Author: limlin
  Contact: limlin95@126.com
  Datetime: 2021/9/6 22:12
  Software: PyCharm
  Profile:
"""
def text_filter(text: str) -> str:
    """
    文本过滤器：过滤掉文本数据中的标点符号和其他特殊字符
    :param text: 待过滤的文本
    :return: 过滤后的文本
    """
    result = str()
    for t in text:
        if t.isalnum():
            if t.isalpha():
                t = t.lower()
            result += str(t)
    return result

def slide_word(text: str, l: int = 5) -> list:
    """
    滑动取词器
    Input: text='abcd',l=2
    Output: ['ab','bc','cd']
    :param text: 过滤后的文本 （只包含小写数字/字母）
    :param l: 滑动窗口长度，默认为 5
    :return:
    """
    tf = text_filter(text)
    result = list()
    if len(tf) <= l:
        result.append(tf)
        return result
    for i in range(len(tf)):
        word = tf[i:i + l]
        if len(word) < l:
            break
        result.append(word)
    return result


if __name__ == '__main__':
    banner = 'abcdefghigkLMN*^%$*   \r\n)021'
    print(slide_word(banner))