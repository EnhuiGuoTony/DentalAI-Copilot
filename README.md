# DentalAI-Copilot

## OpenRouter free-model setup

The chat service uses OpenRouter's free-model router by default. Create a local
`backend/.env` from `backend/.env.example`, then set the following values:

```env
LLM_PROVIDER=openrouter
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openrouter/free
OPENROUTER_API_KEY=your_openrouter_api_key
MOCK_LLM=false
```

`OPENROUTER_API_KEY` is ignored by Git; never put it in frontend code. Restart
the backend after changing this file, then use **Connect model** in the web UI.
