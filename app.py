import os

from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains.question_answering import load_qa_chain

from langchain.vectorstores import Chroma
import pandas as pd
import json
import streamlit as st



os.environ["DASHSCOPE_API_KEY"] = 'sk-38e455061c004036a70f661a768ba779'
DASHSCOPE_API_KEY='sk-38e455061c004036a70f661a768ba779'


embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db_modelY", embedding_function=embeddings)

# vectorstore.as_retriever(search_kwargs={'k': 1})
# vectorstore.as_retriever(search_type="mmr")
# vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={'k': 3,'score_threshold': 0.1})



# print(len(vectorstore.get(limit=1)))
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="brand",
        description="汽车品牌",
        type="string",
    ),
    AttributeInfo(
        name="model",
        description="车型",
        type="string",
    ),
    AttributeInfo(
        name="name",
        description="具体车型名称",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="上市年份",
        type="integer",
    ),
    AttributeInfo(
        name="price", 
        description="售价", 
        type="string"
    )
]
document_content_description = "汽车车型的用户评价"
llm = OpenAI(temperature=0)


retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)
# ,enable_limit=True

# retriever=SelfQueryRetriever(search_kwargs={"k":3})
# retriever.from_llm(llm=llm,vectorstore=vectorstore,document_content_description=document_content_description,metadata_field_info=metadata_field_info,verbose=True,enable_limit=True)



#✔️ 增加name属性
# print(retriever.get_relevant_documents(query="李光L9,2.0T自动优点"))

# filter 2.0T自动 丢失
# print(retriever.get_relevant_documents(query="我最近考虑买车，目前关注李光L9这款车，请介绍一下2.0T自动优点和缺点"))

# 这个可以，三个过滤条件
# print(retriever.get_relevant_documents(query="请介绍李光L9纯电动,这款车的缺点"))
# 四个过滤条件就不行了，目前最多只能三个过滤条件??????结论不扎实，纯电动这个过滤条件丢失了
# print(retriever.get_relevant_documents(query="请介绍李光L9纯电动,这款车的缺点"))


# ✔️ 可以找出缺点
# print(retriever.get_relevant_documents(query="李光L9的缺点"))


# ✔️ 全部找出来，把优点排前面，缺点排后面
# print(retriever.get_relevant_documents(query="丰田卡罗拉优点,2020年上市"))

# print(retriever.get_relevant_documents(query="驾驶者之车",metadata={"brand": '理想'}))

# This example only specifies a relevant query
# ✔️
# print(retriever.get_relevant_documents("大众高尔夫的优点"))
# ✔️ 
# print(retriever.get_relevant_documents("2020年之后上市的宝马"))
# print(retriever.get_relevant_documents("2015年之后上市的宝马"))

# 2.检索生成结果
def retrieve_info(query):
    return retriever.get_relevant_documents(query=query)

# 3.设置LLMChain和提示
llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k-0613')

template = """
    你是一名掌握了全部汽车用户真实使用评价内容的智能回复机器人。
    我将发送给你一位客户关于汽车使用、购买建议、与其他品牌车型对比等方面的问题。
    客户希望你在真实车主评价的基础上，归纳总结形成一句结论性的内容，发送给这位客户，并遵循以下所有规则。
    1/ 在篇幅、语气、逻辑论证和其他细节方面，答复应与尽可能的给人专业的感觉，如实客观的表达问题的答案，不要增加你自己的幻觉。
    2/ 如果在真实车主评价内容中没有这个问题的相关答案，请回答：“很抱歉，基于真实车主的口碑数据，我暂时不能给出您这个问题的答案。“
    {message}
    以下是针对这个问题，真实车主评价内容：
    {best_practice}
    请为这个客户返回最符合问题的最佳回复内容：

    所有回复均为中文
"""
prompt=PromptTemplate(
    input_variables=["message","best_practice"],
    template=template
)



st.set_page_config(page_title="汽车口碑GPT",page_icon="🚗")

chain=LLMChain(llm=llm,prompt=prompt)
# 4.检索生成结果
def generate_response(message):
    best_practice = retrieve_info(message)

    # st.markdown(f'<small style="color: grey;">向量召回内容：{best_practice}</small>', unsafe_allow_html=True)
    # 获取每个 Document 对象中的 page_content 属性，并将其内容组合为一个字符串
    best_practice_text = "<br>".join([doc.page_content for doc in best_practice])

    # 在页面上以较小的字体打印 best_practice_text 变量的内容，并设置颜色为淡灰色
    st.markdown(f'<small style="color: #aaaaaa;">召回内容：<br>{best_practice_text}</small>', unsafe_allow_html=True)




    print('message：',message)
    print('向量召回内容Len：',len(best_practice))
    print('向量召回内容：',best_practice)

    print('')
    print('')
    print('')
    print('')
    print('')
    print('')
    print('')

    # chain_qw = load_qa_chain(llm=llm_qwen, chain_type="stuff",prompt=prompt)
    # chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=prompt)
    # response=chain({"input_documents": best_practice, "question": message}, return_only_outputs=True)



    # response=chain_qw({"input_documents": best_practice, "question": message}, return_only_outputs=True)
    # response=chain.run(input_documents=best_practice, question=message)
    response = chain.run(message=message,best_practice=best_practice)
    return response

# message='特斯拉ModelY的后备箱可以放下自行车么？'
# message='特斯拉ModelY的后备箱可以放下冰箱么？'

# 很抱歉，基于真实车主的口碑数据，我暂时不能给出您这个问题的答案。
# message='特斯拉ModelY四驱能越野么？'


# message='特斯拉Model Y的后备箱可以放下自行车么？'
# print(generate_response(message))

# 5.创建一个应用使用streamlit框架
def main():

    st.header("汽车口碑GPT 🚗")

    message = st.text_area("问问我吧：我知道关于特斯拉ModelY的一切问题：冬天续航衰减多少？后备箱能放下自行车么？")

    if message:
        
        result_placeholder = st.empty()  # 创建一个空位，用于显示临时消息
        result_placeholder.write("正在生成回复内容，请稍后...")

        result = generate_response("特斯拉ModelY"+message)
        
        st.info(result)
        result_placeholder.empty()  # 清空临时消息


if __name__ == "__main__":
    main()
