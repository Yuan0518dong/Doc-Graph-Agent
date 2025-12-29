import json
from openai import OpenAI
import sys
import os

# 确保能找到 src.retrieval
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.retrieval.graph_engine import GraphRetriever

# === 配置 ===
API_KEY = "sk-5f460d116b4243f498d356b5fb052fa5"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"


class GraphRAGAgent:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.retriever = GraphRetriever()

    def close(self):
        self.retriever.close()

    def extract_keywords(self, question: str) -> list:
        """
        利用 DeepSeek 将自然语言问题转化为搜索关键词
        """
        prompt = f"""
        请从用户的问题中提取 2-3 个核心搜索关键词（实体）。
        问题："{question}"

        要求：
        1. 只输出关键词列表，不要包含其他文字。
        2. 格式必须是 JSON 列表，例如：["Transformer", "Attention", "Google"]
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            # 简单的清洗逻辑
            content = content.replace("```json", "").replace("```", "").strip()
            keywords = json.loads(content)
            return keywords
        except Exception as e:
            print(f"⚠️ 关键词提取失败: {e}，将使用备用策略。")
            return question.split()[:2]

    def chat(self, question: str):
        print(f"\n🤖 用户提问: {question}")

        # 1. 思考关键词
        keywords = self.extract_keywords(question)
        print(f"🔍 思考出的搜索词: {keywords}")

        # 2. 去图谱里抓数据
        context = self.retriever.query_graph_context(keywords, limit=5)

        if not context:
            print("⚠️ 图谱里没找到相关信息，依靠模型自带知识回答...")
            context = "没有找到相关的背景知识。"
        else:
            # 这里的 replace 是为了打印美观，去掉过多的换行
            print(f"📚 成功检索到背景知识 (前100字): {context[:100].replace(chr(10), ' ')}...")

        # 3. 结合上下文回答
        system_prompt = """
        你是一个基于知识图谱的智能助手。请根据提供的【背景知识】回答用户问题。
        如果背景知识里有答案，请引用它；如果没有，请诚实地说不知道，或者用你自己的知识补充（但要说明）。
        """

        user_prompt = f"""
        【背景知识】：
        {context}

        【用户问题】：
        {question}
        """

        print("⚡ DeepSeek 正在组织语言...")
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            stream=True
        )

        full_response = ""
        print("\n💬 回答:")
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print("\n" + "=" * 50)

        return full_response


# === 测试入口 ===
if __name__ == "__main__":
    agent = GraphRAGAgent()
    # 既然你有很多关于 Agent 的文档，我们试个相关的问题
    agent.chat("什么是智能体（Agent）？它和普通模型有什么区别？")
    agent.close()