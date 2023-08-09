from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
# 1.矢量化数据
loader = CSVLoader(file_path="reshaped_car_data_1000.csv")
documents = loader.load()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents,embeddings)
# 2.做相似性搜索
def retrieve_info(query):
    similar_response = db.similarity_search(query,k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    print(page_contents_array)
    return page_contents_array
# custom_prompt = """
#     我想合作或定制服务，怎么联系？
# """
# results=retrieve_info(custom_prompt)
# print(results)

# 3.设置LLMChain和提示
llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k-0613')
template = """
    你是一名经验丰富的汽车口碑知识库客服机器人.
    我将与你分享一位客户的用车方面的问题，你将给出一个最佳答案。
    根据你的经验给出最佳的答案，并发送给这位潜在客户，并遵循以下所有规则。.
    1/ 在篇幅、语气、逻辑论证和其他细节方面，答复应与过去的最佳做法非常相似，甚至完全相同。
    2/ 如果与最佳实践无关，则应尽量模仿最佳实践的风格，以传达潜在客户的信息。
    以下是我从潜在客户那里收到的信息：
    {message}
    以下是此类用车问题，的真实车主反馈
    {best_practice}
    请给出你认为最佳的问题答案：
"""
prompt=PromptTemplate(
    input_variables=["message","best_practice"],
    template=template
)
chain=LLMChain(llm=llm,prompt=prompt)
# 4.检索生成结果
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message,best_practice=best_practice)
    return response


# response=generate_response(message)
# print(response)
# 5.创建一个应用使用streamlit框架
def main():
    st.set_page_config(
        page_title="汽车口碑PGT",page_icon="🚗")

    st.header("汽车口碑PGT 🚗")
    message = st.text_area("例：丰田卡罗拉2021款的高速表现如何？")

    if message:
        st.write("正在生成回复内容，请稍后...")

        result = generate_response(message)
        
        st.info(result)
        st.write("")

if __name__ == "__main__":
    main()