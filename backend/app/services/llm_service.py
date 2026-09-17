import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from app.core.config import get_settings
from app.schemas.chat import ChatMessage


class LlmService:
    """统一聊天模型的配置与实例化，隔离 OpenAI 兼容接口和 Google 接口差异。

    PmsAgentService 使用这里的模型工厂执行工具循环；chat 方法提供普通单次对话，
    当前 /chat 路由实际调用 PmsAgentService，不直接使用下面的 chat 方法。
    """
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        # 只判断本地配置，不验证密钥是否有效、网络是否连通或模型是否支持工具调用。
        return bool(self._api_key()) and not self.settings.mock_llm

    def connect(self) -> tuple[bool, str]:
        """检查配置并尝试构造客户端，不发起模型请求，因此不代表远端连接测试成功。"""
        if self.settings.mock_llm:
            return True, "Mock LLM is ready."
        if not self._api_key():
            return False, "No API key is configured for the selected provider."
        try:
            self._chat_model()
        except (TypeError, ValueError) as error:
            return False, str(error)
        return True, "Model configuration is ready."

    def chat(self, message: str, history: list[ChatMessage] | None = None, patient_context: dict | None = None) -> str:
        """一次性发送系统指令、历史与可选数据库上下文；这里不注册工具或自动检索。"""
        if not self.enabled:
            return (
                "Mock response: OpenRouter is not enabled yet. Set MOCK_LLM=false "
                "and configure OPENROUTER_API_KEY to receive real model responses."
            )

        # 系统指令保持英文；提示词表达约束，但本身不能保证模型绝不产生幻觉。
        messages = [
            SystemMessage(
                content=(
                    "Your identity is DentalAI Copilot, a dental clinical support agent. "
                    "You help dental professionals organize case information, explain dental concepts, "
                    "and prepare concise clinical drafts for licensed clinician review. "
                    "Never identify yourself as Nex, Nex-AGI, OpenRouter, a language model, or any "
                    "other provider or agent. If asked who you are, say only that you are DentalAI Copilot, "
                    "a dental clinical support agent. Answer in the user's language. Do not present any "
                    "medical information as a final diagnosis; recommend clinician review where appropriate. "
                    "Reply in the same language as the user's latest message. "
                    " When CHART CONTEXT is supplied, use it as the authoritative source for patient counts, "
                    "names, chart facts, and notes. Answer the user's question in detail from that context; "
                    "never invent patient data. Say clearly when the context has no matching information."
                )
            )
        ]
        # 客户端只能提供用户/助手历史，系统角色由服务端控制。
        # 这是消息角色边界，并不等同于对用户文本中的提示词注入做了完整防护。
        for item in history or []:
            if item.role == "user":
                messages.append(HumanMessage(content=item.content))
            elif item.role == "assistant":
                messages.append(AIMessage(content=item.content))

        # 上下文序列化为 JSON 后附到本轮用户消息，并不会变成模型权重或长期记忆。
        # 本方法不自行脱敏，调用者需在进入模型边界前处理敏感信息。
        context_suffix = ""
        if patient_context is not None:
            context_suffix = f"\n\nCHART CONTEXT (read-only database result):\n{json.dumps(patient_context, default=str)}"
        messages.append(HumanMessage(content=message + context_suffix))
        result = self._chat_model().invoke(messages)
        return str(result.content)

    def provider_label(self) -> str:
        provider = self.settings.llm_provider.lower()
        if provider == "google":
            return f"google:{self.settings.google_model}"
        if provider == "openrouter":
            return f"openrouter:{self.settings.llm_model}"
        return f"openai:{self.settings.llm_model}"

    def _api_key(self) -> str:
        """优先读取提供方专用密钥，缺失时回退到通用密钥；密钥只留在后端配置中。"""
        provider = self.settings.llm_provider.lower()
        if provider == "google":
            return self.settings.google_api_key or self.settings.llm_api_key
        if provider == "openrouter":
            return self.settings.openrouter_api_key or self.settings.llm_api_key
        return self.settings.llm_api_key

    def _chat_model(self):
        """按提供方创建 LangChain 聊天客户端；真正发出请求发生在 invoke 等调用时。"""
        # 较低 temperature 用于减少回答随机性，但不保证确定性或事实正确性。
        # OpenRouter 使用 OpenAI 兼容协议，通过 base_url 切换服务入口。
        provider = self.settings.llm_provider.lower()
        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=self.settings.google_model,
                api_key=self._api_key(),
                temperature=0.2,
            )
        if provider in {"openai", "openrouter"}:
            return ChatOpenAI(
                model=self.settings.llm_model,
                api_key=self._api_key(),
                base_url=self.settings.llm_base_url,
                temperature=0.2,
            )
        raise ValueError(f"Unsupported LLM_PROVIDER: {self.settings.llm_provider}")
