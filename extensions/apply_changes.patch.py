import re
import time
from typing import List, Dict

from openai import OpenAI

from common.api import ExtensionAPI
from common.diff_lines import get_matches

META_TAG = 'metadata'


def get_system_prompt(model: str):
    system_prompt = f"""
You are an intelligent programmer, powered by {model}. You are happy to help answer any questions that the user has (usually they will be about coding).

You are given a file and a code edit suggested by an AI and the original file. Respond with the full contents of the file with the changes applied. Do not make any other changes to the file. Respond directly with the full contents of the file.
""".strip()

    return system_prompt


def get_system_prompt_alt1(model: str):
    system_prompt = f"""
You are an intelligent programmer, powered by {model}. You are happy to help answer any questions that the user has (usually they will be about coding).

You are given a file and a code edit suggested by an AI. You need to find where to exactly apply the edit in the file. Give the code line numbers to apply the edit. Only respond with the start and end line `[[n, m]]` where $n <= m$. The lines $n, n + 1, n + 2, ..., m - 1$ will get replaced with **all** the lines in the suggested edit. If $n = m$ then the edit is inserted after line $n$ and no lines get replaced.
""".strip()

    return system_prompt


def _format_code_block(content: str, lineno: bool = False) -> str:
    if not lineno:
        return f"```\n{content}\n```"
    else:
        lines = content.split('\n')
        numbered_lines = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines)]
        content = '\n'.join(numbered_lines)
        return f"```\n{content}\n```"


def _format_section(title: str, content: str) -> str:
    return f"## {title}\n\n{content}"


def _strip_backticks(text: str) -> str:
    match = re.search(r"```(?:[\w-]+\n)?([\s\S]*?)```", text)
    if match:
        return match.group(1).strip("\n")
    return text
    

def call_llm(api: ExtensionAPI, model: str, messages: List[Dict[str, str]], content: str):
    """Streams responses from the LLM and sends them to the chat UI in real-time."""

    start_time = time.time()

    client = OpenAI(api_key=api.api_key, base_url=api.api_url)

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.8,
        top_p=0.8,
    )

    prompt_tokens = None
    buffer, patch = '', ''
    for chunk in stream:
        delta = chunk.choices[0].delta

        if delta.content:
            buffer += delta.content
            patch += delta.content

        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
    
    cleaned_patch = _strip_backticks(patch)
    matches, cleaned_patch = get_matches(content, cleaned_patch)
    
    api.apply_diff(cleaned_patch, matches)

    elapsed = time.time() - start_time
    meta_data = f'time: {elapsed:.2f}s'
    if prompt_tokens is not None:
        meta_data += f' prompt: {prompt_tokens :,} model: {model}'


def extension(api: ExtensionAPI):
    """Main extension function that handles chat interactions with the AI assistant."""

    # Default model mappings
    model_mappings = {
        'v3': 'deepseek/deepseek-chat-v3-0324',  # Regular deepseek model
        'r1': 'deepseek/deepseek-r1-0528',  # Deepseek model with 64k context
        'o3': 'openai/o3',  # OpenAI model
        'devstral': 'mistralai/devstral-small',
        'mistral3b': 'mistralai/ministral-3b',
    }

    prompt = api.prompt.rstrip()  # No left strip for indentation
    
    api.log(f'{api.edit_file.path}/{api.current_file.path}')
    
    if api.edit_file.path == '':
        content = api.current_file_content
    if api.edit_file.path == api.current_file.path:
        content = api.current_file_content
    else:
        content = api.edit_file.get_content()
        
    api.log(f'{api.edit_file.path}/{api.current_file.path}')

    sections = [
        _format_section("Edit Suggested by AI", _format_code_block(prompt, False)),
        _format_section("File to Edit", _format_code_block(content, False)),
        _format_section("Edit Suggested by AI (same as above)", _format_code_block(prompt, False)),
        'Apply the suggested edit to the file.',
        # 'Give the code line numbers to apply the edit. Only respond with the start and end line `[[n, m]]` where $n <= m$. The lines $n, n + 1, n + 2, ..., m - 1$ will get replaced with **all** the lines in the suggested edit. Not that the line $m$ does not get replacd.',
        # 'The suggested edit may contain lines before and after the actually changed lines, including comments or blank lines. These are given to locate where the change should apply. These should also get replaced such that after the edit is applied there are no missing lines or repeated lines.',
    ]

    model = model_mappings['mistral3b']

    prompt = '\n\n'.join(sections)
    api.log(prompt)
    messages = [
        {'role': 'system', 'content': get_system_prompt(model)},
        {'role': 'user', 'content': prompt},
    ]

    api.log(f'messages {len(messages)}')
    api.log(f'prompt {api.prompt}')
    # api.log(context)

    call_llm(api, model, messages, content)