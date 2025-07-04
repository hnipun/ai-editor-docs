from typing import Dict, List

from openai import OpenAI

from common.api import ExtensionAPI


def get_system_prompt(model: str):
    system_prompt = f"""
You are an intelligent programmer, powered by {model}. You are happy to help answer any questions that the user has (usually they will be about coding).

1. When the user is asking for edits to their code, please output a simplified version of the code block that highlights the changes necessary and adds comments to indicate where unchanged code has been skipped. For example:

```language:path/to/file
// ... existing code ...
{{ edit_1 }}
// ... existing code ...
{{ edit_2 }}
// ... existing code ...
```

The user can see the entire file, so they prefer to only read the updates to the code. Often this will mean that the start/end of the file will be skipped, but that's okay! Rewrite the entire file only if specifically requested. Always provide a brief explanation of the updates, unless the user specifically requests only the code.

These edit codeblocks are also read by a less intelligent language model, colloquially called the apply model, to update the file. To help specify the edit to the apply model, you will be very careful when generating the codeblock to not introduce ambiguity. You will specify all unchanged regions (code and comments) of the file with â// â¦ existing code â¦â comment markers. This will ensure the apply model will not delete existing unchanged code or comments when editing the file. You will not mention the apply model.

2. Do not lie or make up facts.

3. Format your response in markdown.

4. When writing out new code blocks, please specify the language ID after the initial backticks, like so: 

```python
{{ code }}
```

5. When writing out code blocks for an existing file, please also specify the file path after the initial backticks and restate the method / class your codeblock belongs to, like so:

```language:some/other/file
function AIChatHistory() {{
    ...
    {{ code }}
    ...
}}
```
""".strip()

    return system_prompt


def _format_code_block(content: str) -> str:
    return f"```\n{content}\n```"


def _format_section(title: str, content: str) -> str:
    return f"## {title}\n\n{content}"


def build_context(api: ExtensionAPI) -> str:
    """Builds the context string from the current file and selection."""
    
    context = []

    if api.opened_files:
        opened_files = [f'Path: `{f.path}`\n\n' + _format_code_block(f.get_content()) for f in api.opened_files]

        context.append(_format_section("Other relevant files", "\n\n".join(opened_files)))
        
    api.push_to_chat(content=f"\n<metadata> Opened Files: {[f.path for f in api.opened_files]} </metadata>\n")

    context.append(
        _format_section("Current File",
                        f"Here is the file I'm looking at (`{api.current_file.path}`):\n\n" +
                        _format_code_block(api.current_file_content))
    )
    
    api.push_to_chat(content=f"\n<metadata> Current File: {api.current_file.path} </metadata>\n")

    if api.selection and api.selection.strip():
        context.append(
            _format_section("Selection",
                            "This is the code snippet that I'm referring to\n\n" +
                            _format_code_block(api.selection))
        )

    return "\n\n".join(context)

def call_llm(api: ExtensionAPI, model: str, messages: List[Dict[str, str]]):
    """Streams responses from the LLM and sends them to the chat UI in real-time."""
    
    client = OpenAI(api_key=api.api_key, base_url=api.api_url)

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=1,
        top_p=1,
        extra_body={
        "reasoning": {          
            "max_tokens": 1500 
        }
    },
    )
    
    thinking = False
    for chunk in stream:
        delta = chunk.choices[0].delta
        if getattr(delta, 'reasoning', None):
            if not thinking:
                api.push_to_chat(content="<collapse>")
                thinking = True
            api.push_to_chat(content=delta.reasoning)
        
        if delta.content:
            if thinking:
                api.push_to_chat(content="</collapse>")
                thinking = False
            api.push_to_chat(content=delta.content)
            
        if chunk.usage is not None:    
             prompt_tokens = chunk.usage.prompt_tokens
             api.push_to_chat(content=f"\n<metadata> number of tokens used in prompt: {prompt_tokens} model: {model} </metadata>\n")

    api.terminate_chat()


def extension(api: ExtensionAPI):
    """Main extension function that handles chat interactions with the AI assistant."""
    
    # need to use a open-router model name (https://openrouter.ai/rankings/programming?view=week)
    # model = "openai/o3"
    # model = 'anthropic/claude-opus-4'
    model = 'anthropic/claude-sonnet-4'
    
    context = build_context(api)

    prompt = api.prompt.strip()

    if prompt.startswith('/'):
        prompt = prompt[1:].strip()
        messages = [
            {'role': 'system', 'content': get_system_prompt(model)},
            *[m.to_dict() for m in api.chat_history],
            {'role': 'user', 'content': api.prompt},
        ]
    else:
        messages = [
            {'role': 'system', 'content': get_system_prompt(model)},
            {'role': 'user', 'content': context},
            *[m.to_dict() for m in api.chat_history],
            {'role': 'user', 'content': api.prompt},
        ]

    api.log(f'messages {len(messages)}')
    api.log(f'prompt {api.prompt}')
    terminal_snapshot = "\n".join(api.terminal_snapshot)
    api.log(f'## Terminal {terminal_snapshot}')

    call_llm(api, model, messages)