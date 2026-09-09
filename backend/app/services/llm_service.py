import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.config import get_settings
from app.schemas.agent import AgentOutput
from app.schemas.chat import ChatMessage


class ClinicalDraftInput(BaseModel):
    question: str
    xray_findings: list[dict]
    evidence: list[dict]
    structured_facts: list[dict] = []


class LlmService:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        return bool(self._api_key()) and not self.settings.mock_llm

    def generate_clinical_draft(self, payload: ClinicalDraftInput) -> AgentOutput:
        if not self.enabled:
            raise RuntimeError("LLM is disabled. Set MOCK_LLM=false and configure the selected provider API key.")

        model = self._chat_model()
        structured_model = model.with_structured_output(AgentOutput)
        result = structured_model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a dental clinical AI assistant. Produce concise, evidence-backed "
                        "drafts for licensed clinician review. Never present the output as a final diagnosis."
                    )
                ),
                HumanMessage(
                    content=(
                        "Generate a structured clinical draft from this case context:\n"
                        f"{json.dumps(payload.model_dump(), default=str)}"
                    )
                ),
            ]
        )
        if not isinstance(result, AgentOutput):
            return AgentOutput.model_validate(result)
        return result

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

    def chat(self, message: str, history: list[ChatMessage] | None = None) -> str:
        if not self.enabled:
            return (
                "Mock response: OpenRouter is not enabled yet. Set MOCK_LLM=false "
                "and configure OPENROUTER_API_KEY to receive real model responses."
            )

        messages = [
            SystemMessage(
                content=(
                    "You are DentalAI Copilot, a practical AI engineering portfolio assistant. "
                    "Answer clearly and concisely. When discussing medical content, avoid claiming "
                    "to provide a final clinical diagnosis."
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

        messages.append(HumanMessage(content=message))
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
