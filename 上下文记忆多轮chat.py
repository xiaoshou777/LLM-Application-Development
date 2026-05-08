from openai import OpenAI, APIError, APIConnectionError, AuthenticationError
from dotenv import load_dotenv
import os
import logging
from datetime import datetime

# ==================== 日志配置（自动打印时间、级别、信息） ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 加载 .env 文件
load_dotenv()

# 全局维护上下文历史（核心：实现记忆）
conversation_history = [
    {"role": "system",
     "content": "你是大模型开发应用专家，我正学习该领域知识、寻求相关工作，你负责解答我提出的各类问题，语言尽量简洁，必要时需提供具体实现代码。"}
]

# ==================== 上下文长度限制 ====================
MAX_HISTORY_ROUNDS = 5  # 最多保留最近 5 轮对话


def trim_conversation_history():
    """修剪上下文：防止 token 超出模型上限"""
    global conversation_history
    system_msg = conversation_history[:1]
    history_msgs = conversation_history[1:][-(MAX_HISTORY_ROUNDS * 2):]
    conversation_history = system_msg + history_msgs


def llm_chat_with_memory(
        prompt: str,
        api_key: str,
        base_url: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024
) -> str:
    """
    带记忆+异常处理+日志打印的大模型对话
    :return: 模型回答 / 错误提示
    """
    if not all([api_key, base_url, model_name]):
        logging.error("环境变量缺失：请检查 API_KEY / BASE_URL / MODEL_ID 是否配置")
        return "配置错误：请检查 .env 文件"

    try:
        # 日志：用户输入
        logging.info(f"用户提问：{prompt}")

        # 初始化客户端
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 加入用户消息
        conversation_history.append({"role": "user", "content": prompt})
        trim_conversation_history()

        # 发送请求
        logging.info("正在请求模型生成回答...")
        response = client.chat.completions.create(
            model=model_name,
            messages=conversation_history,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # 获取回答
        assistant_reply = response.choices[0].message.content.strip()
        logging.info(f"模型回复：{assistant_reply}")

        # 保存到记忆
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    except AuthenticationError:
        logging.error("认证失败：API Key 错误或无效")
        return "错误：API Key 无效，请检查配置"

    except APIConnectionError:
        logging.error("连接失败：网络错误或 BASE_URL 不可用")
        return "错误：无法连接到模型服务，请检查网络或接口地址"

    except APIError as e:
        logging.error(f"模型接口报错：{str(e)}")
        return f"模型服务异常：{e.message}"

    except Exception as e:
        logging.error(f"未知错误：{str(e)}", exc_info=True)
        return f"程序出现未知错误：{str(e)}"


if __name__ == "__main__":
    print("=== 支持上下文记忆的多轮聊天机器人 ===")
    print("输入 'exit' 退出，输入 'clear' 清空记忆\n")

    # 检查环境变量
    required_vars = ["API_KEY", "BASE_URL", "MODEL_ID"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logging.error(f"缺少环境变量：{', '.join(missing)}")
        print("请先配置 .env 文件再运行！")
        exit(1)

    while True:
        question = input("输入：").strip()

        if not question:
            print("输出：请输入有效内容\n")
            continue

        if question.lower() == "exit":
            print("输出：再见！")
            break

        if question.lower() == "clear":
            conversation_history = [conversation_history[0]]
            logging.info("已清空对话记忆")
            print("输出：已清空上下文记忆！\n")
            continue

        # 调用对话函数
        res = llm_chat_with_memory(
            prompt=question,
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
            model_name=os.getenv("MODEL_ID")
        )

        print("输出：", res, "\n")