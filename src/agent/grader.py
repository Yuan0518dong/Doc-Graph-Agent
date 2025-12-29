from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# === 配置 ===
API_KEY = "sk-5f460d116b4243f498d356b5fb052fa5" # 保持一致
BASE_URL = "https://api.deepseek.com"

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model="deepseek-chat",
    temperature=0
)

# 定义输出结构（用 Pydantic 保证格式稳定）
class Grade(BaseModel):
    """二元评分：文档是否相关"""
    binary_score: str = Field(description="相关性评分，必须是 'yes' 或 'no'")

# 使用 .with_structured_output 让模型强制输出 JSON
structured_llm_grader = llm.with_structured_output(Grade)

system = """你是一个评分员，负责评估检索到的文档与用户问题的相关性。
如果文档包含关键词或语义上与问题相关，评为 'yes'，否则评为 'no'。
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "用户问题: {question}\n\n检索文档: {document}\n\n相关吗(yes/no)?"),
    ]
)

grader_chain = grade_prompt | structured_llm_grader

def grade_document(question: str, document: str) -> str:
    """
    调用大模型进行评分
    """
    # 快速检查：如果文档是空的或者报错信息，直接不通过
    if "没有找到" in document or "工具报错" in document:
        return "no"
        
    try:
        score = grader_chain.invoke({"question": question, "document": document})
        return score.binary_score
    except Exception as e:
        print(f"⚠️ 评分出错，默认视为相关: {e}")
        return "yes"