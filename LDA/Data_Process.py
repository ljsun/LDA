# coding = utf-8

import pandas as pd
import re
import jieba.posseg as pseg


def data_process(path=None):

    flag = ['t', 'nr', 'a', 'd', 'f', 'z', 'r', 'm', 'q', 'p', 'c', 'u', 'e', 'y', 'o', 'h', 'k', 'l', 'w']
    stopword = []
    with open('./input/stopword.txt') as fs:
        for single_stopword in fs.readlines():
            stopword.append(single_stopword.strip('\n').strip(' '))

    def clean_weibo_text(text):

        # flag = ['t', 'x', 'r']
        # 去除“看不见”的字符，例如\n、\t
        text = text.replace(r'\s', " ")
        # 去除标点、数字、英文字母
        text = re.sub(r"[,，。“”！；：、#\da-zA-Z]", "", text)
        # 去除\u200b
        text = re.sub('\u200b', '', text)
        # 去除网页链接，图片来源等字样
        text = re.sub(r'[网页链接]|[图片来源]', '', text)

        return text
    df = pd.read_csv(open('./input/weibo_data.csv', 'rU'))
    df = df[['comment_num', 'content', 'praise_num']].dropna()
    content = df['content']
    df['content'] = content.apply(lambda s: clean_weibo_text(s))
    content_list = df['content'].values
    comment_num_list = df['comment_num'].values
    praise_num_list = df['praise_num'].values

    zipped = list(zip(content_list, comment_num_list, praise_num_list))
    LDA_texts = dict()
    HLDA_texts = dict()
    unique_word = set()
    index = 0
    for content, comment_num, praise_num in zipped:
        text = []
        for text_word in pseg.cut(content):
            if text_word.word not in stopword and len(text_word.word.strip()) > 1 and text_word.flag not in flag:
                text.append(text_word.word)
                unique_word.add(text_word.word)
        if len(text) != 0:
            LDA_texts[index] = text
            HLDA_texts[index] = [text, int(comment_num), int(praise_num)]
            index += 1

    return LDA_texts, HLDA_texts, unique_word

if __name__ == '__main__':
    LDA_texts, HLDA_texts, unique_word = data_process()
    print(HLDA_texts)

