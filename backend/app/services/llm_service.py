import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from app.core.config import get_settings
from app.schemas.chat import ChatMessage


class LlmService:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        return bool(self._api_key()) and not self.settings.mock_llm

    def connect(self) -> tuple[bool, str]:
        """Prepare provider configuration without a model invocation or token use."""
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
        if not self.enabled:
            return (
                "Mock response: OpenRouter is not enabled yet. Set MOCK_LLM=false "
                "and configure OPENROUTER_API_KEY to receive real model responses."
            )

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
        # The API schema only permits prior user/assistant turns. The server
        # owns the system prompt, preventing irrelevant client context.
        for item in history or []:
            if item.role == "user":
                messages.append(HumanMessage(content=item.content))
            elif item.role == "assistant":
                messages.append(AIMessage(content=item.content))

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
        provider = self.settings.llm_provider.lower()
        if provider == "google":
            return self.settings.google_api_key or self.settings.llm_api_key
        if provider == "openrouter":
            return self.settings.openrouter_api_key or self.settings.llm_api_key
        return self.settings.llm_api_key

    def _chat_model(self):
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
