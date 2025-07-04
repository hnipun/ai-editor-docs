import json
from pathlib import Path
from typing import Dict, List, Optional

TAGS = {
    'think': 'collapse',
    'meta': 'metadata'
}


class File:
    def __init__(self, path: str, repo_path: str):
        self.path: str = path
        self._fs_path = Path(f'{repo_path}/{path}')

    def suffix(self) -> str:
        return self._fs_path.suffix

    def exists(self) -> bool:
        return self._fs_path.is_file()

    def get_content(self) -> str:
        with open(self._fs_path, 'r') as f:
            return f.read()


class Message:
    def __init__(self, **kwargs):
        self.role: str = kwargs["role"]
        self.content: str = kwargs["content"]

    def to_dict(self) -> Dict[str, str]:
        return {'role': self.role, 'content': self.content}


class ExtensionAPI:
    """Main API class that provides access to editor state and operations.

    Attributes:
            repo: List of all files in the repository
            current_file: Currently focused file
            current_file_content: Content of current file
            opened_files: List of currently opened files
            selection: Currently selected text (if any)
            clip_board: Current clipboard content (if any)
            cursor_row: Current cursor row position
            cursor_column: Current cursor column position
            chat_history: List of previous chat messages
            prompt: Current user prompt
            api_key: API key for LLM services
    """

    repo: List[File]
    current_file: File
    edit_file: Optional[File]
    current_file_content: str
    opened_files: List[File]
    selection: Optional[str]
    clip_board: Optional[str]
    cursor_row: int
    cursor_column: int
    chat_history: List[Message]
    terminal_history: Optional[str]
    terminal_snapshot: Optional[List[str]]
    prompt: str
    api_key: str
    api_url: str

    _blocks: List[str]

    def load(self, **kwargs):
        self.current_file_content = kwargs['current_file_content']
        self.selection = kwargs['selection']
        self.cursor_row = kwargs['cursor_row']
        self.cursor_column = kwargs['cursor_column']
        self.api_key = kwargs['api_key']
        self.api_url = kwargs['api_url']
        self.prompt = kwargs['prompt']
        self.terminal_history = kwargs.get('terminal_history', None)
        self.terminal_snapshot = kwargs.get('terminal_snapshot', None)
        self.repo_path = kwargs['repo_path']
        self.current_file = File(kwargs['current_file'], self.repo_path)
        self.edit_file = File(kwargs['edit_file'], self.repo_path) if 'edit_file' in kwargs else None

        self.repo_files = [File(p, self.repo_path) for p in kwargs['repo']]
        self.opened_files = [File(p, self.repo_path) for p in kwargs['opened_files']]

        self.chat_history = [Message(**m) for m in kwargs['chat_history']]

        self._blocks = []

        return self

    def _dump(self, method: str, **kwargs):
        assert 'method' not in kwargs
        kwargs['method'] = method

        print(json.dumps(kwargs), flush=True)

    def push_to_chat(self, content: str):
        """Send content to be displayed in the chat UI."""
        self._dump('push_chat', content=content)

    def start_block(self, type_: str):
        """Start a block of type `type`. `type_` can be `meta` or `think`."""

        tag = TAGS[type_]

        self.push_to_chat(f"<{tag}>")
        self._blocks.append(tag)

    def end_block(self):
        """
        End the current block
        """
        assert len(self._blocks) > 0

        tag = self._blocks.pop(-1)

        self.push_to_chat(f'</{tag}>')

    def push_block(self, type_: str, content: str):
        """
        Send a block of type `type`. `type_` can be `meta` or `think`
        """
        self.start_block(type_)
        self.push_to_chat(content)
        self.end_block()

    def push_meta(self, content: str):
        """
        Send a meta block.
        """
        self.start_block('meta')
        self.push_to_chat(content)
        self.end_block()

    def apply_diff(self, patch: List[str], matches: List[List[int]]):
        """
        Stream diff-match coordinates to the client UI.

        Args:
            patch: Lines of code in the patch to apply
            matches: list of [row_in_a, row_in_b] pairs returned by
                     extensions.extension_api.diff_lines.get_matches
        """
        self._dump('apply_diff', patch=patch, matches=matches)

    def terminate_chat(self):
        self._dump('terminate_chat')

    def log(self, content: str):
        """Log a debug message, shows in browser console."""
        self._dump('log', content=content)
