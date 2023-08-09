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
    你是一名出色的汽车销售人员.
    我将与你分享一位潜在客户的购车需求信息，你将给出一个最佳答案。
    根据你的经验给出最佳匹配车型，并发送给这位潜在客户，并遵循以下所有规则。.
    1/ 在篇幅、语气、逻辑论证和其他细节方面，答复应与过去的最佳做法非常相似，甚至完全相同。
    2/ 如果与最佳实践无关，则应尽量模仿最佳实践的风格，以传达潜在客户的信息。
    以下是我从潜在客户那里收到的信息：
    {message}
    以下是此类购车需求中，最佳的推荐车型
    {best_practice}
    请给出你认为最佳的推荐车型：
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
        page_title="智能汽车销售客服",page_icon="🔮")

    st.header("智能汽车销售客服 🔮")
    message = st.text_area("请说出你的购车需求和条件：")

    if message:
        st.write("正在生成回复内容，请稍后...")

        result = generate_response(message)
        
        st.info(result)

if __name__ == "__main__":
    # main()