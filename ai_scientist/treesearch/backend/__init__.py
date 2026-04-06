import os

from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

# v3: Default to agent backend. Set AI_SCIENTIST_BACKEND=legacy to use old SDK backends.
_BACKEND_MODE = os.environ.get("AI_SCIENTIST_BACKEND", "agent")

if _BACKEND_MODE == "legacy":
    from . import backend_anthropic, backend_openai

from . import backend_agent


def get_ai_client(model: str, **model_kwargs):
    """
    Get the appropriate AI client based on the model string.

    In v3 (default), returns an AgentBackend instance regardless of model string.
    Set AI_SCIENTIST_BACKEND=legacy to use the old OpenAI/Anthropic SDK clients.

    Args:
        model (str): string identifier for the model to use.
        **model_kwargs: Additional keyword arguments for model configuration.
    Returns:
        An instance of the appropriate AI client.
    """
    if _BACKEND_MODE == "legacy":
        if "claude-" in model:
            return backend_anthropic.get_ai_client(model=model, **model_kwargs)
        else:
            return backend_openai.get_ai_client(model=model, **model_kwargs)

    return backend_agent.get_ai_client(model=model, **model_kwargs)


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    In v3 (default), all queries route through backend_agent.
    Set AI_SCIENTIST_BACKEND=legacy to use old OpenAI/Anthropic backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    # Compile prompt types to markdown strings
    compiled_system = compile_prompt_to_md(system_message) if system_message else None
    compiled_user = compile_prompt_to_md(user_message) if user_message else None

    if _BACKEND_MODE == "legacy":
        # Legacy path: route to SDK backends based on model string
        # Handle models with beta limitations
        # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
        if model.startswith("o1"):
            if system_message and user_message is None:
                user_message = system_message
            elif system_message is None and user_message:
                pass
            elif system_message and user_message:
                system_message["Main Instructions"] = {}
                system_message["Main Instructions"] |= user_message
                user_message = system_message
            system_message = None
            model_kwargs["reasoning_effort"] = "high"
            model_kwargs["max_completion_tokens"] = 100000
            model_kwargs.pop("temperature", None)
            # Recompile after modification
            compiled_system = compile_prompt_to_md(system_message) if system_message else None
            compiled_user = compile_prompt_to_md(user_message) if user_message else None
        else:
            model_kwargs["max_tokens"] = max_tokens

        query_func = backend_anthropic.query if "claude-" in model else backend_openai.query
        output, req_time, in_tok_count, out_tok_count, info = query_func(
            system_message=compiled_system,
            user_message=compiled_user,
            func_spec=func_spec,
            **model_kwargs,
        )
        return output

    # v3 agent path
    output, req_time, in_tok_count, out_tok_count, info = backend_agent.query(
        system_message=compiled_system,
        user_message=compiled_user,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output
