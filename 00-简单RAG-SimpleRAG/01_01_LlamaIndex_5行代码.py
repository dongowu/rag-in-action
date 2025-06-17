"""
注意：运行此代码前，请确保已在环境变量中设置OpenAI API密钥。
在Linux/Mac系统中，可以通过以下命令设置：
export OPENAI_API_KEY='your-api-key'

在Windows系统中，可以通过以下命令设置：
set OPENAI_API_KEY=your-api-key

如果无法取得OpenAI API密钥，也没关系，我们有平替方案，请移步至其它程序。
"""

# 第一行代码：导入相关的库
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.llms.google_genai import  GoogleGenAI
# # 第二行代码：加载数据
#
# llm = GoogleGenAI(
#     api_key="AIzaSyAGlEFOdT0rCxYHokgRW7wK66YLEQgNJO4",
#     # model = "models/gemini-2.0-flash-lite-preview-02-05"
#
# )
# documents = SimpleDirectoryReader(input_files=["D:\\ownCode\\game\\rag-in-action\\90-文档-Data\\黑悟空\\设定.txt"]).load_data()
# # 第三行代码：构建索引
# index = VectorStoreIndex.from_documents(documents,llm = llm)
# # 第四行代码：创建问答引擎
# query_engine = index.as_query_engine()
# # 第五行代码: 开始问答
# print(query_engine.query("黑神话悟空中有哪些战斗工具?"))
#
#
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.deepseek import DeepSeek

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


import logging


logging.basicConfig(level=logging.DEBUG)
llm = DeepSeek(
    model = "deepseek-chat"
)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"

)


documents = SimpleDirectoryReader(input_files=["D:\\ownCode\\game\\rag-in-action\\90-文档-Data\\黑悟空\\设定.txt"]).load_data()
index = VectorStoreIndex.from_documents(documents,  embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm)
print(query_engine.query("黑神话悟空中有哪些战斗工具?",))
