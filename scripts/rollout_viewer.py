# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Optional

import typer
import ujson as json
from rich.highlighter import ReprHighlighter
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, ProgressBar, Select, SelectionList, Static


def check_textual_version():
    # check textual version is equal to 0.52.1
    import textual
    from packaging.version import Version

    if Version(textual.__version__) != Version("0.52.1"):
        raise ImportError(f"Textual version {textual.__version__} is not supported, please pip install textual==0.52.1")


check_textual_version()


class Highlighter(ReprHighlighter):
    highlights = ReprHighlighter.highlights + [
        r"(?P<tag_name>[][\<\>{}()\|（）【】\[\]=`])",
        r"\<\|(?P<tag_name>[\w\W]*?)\|\>",
    ]


def get_jsonl_file_num(dir: Path, suffix: str = ".jsonl") -> int:
    return sum(1 for _ in dir.glob(f"*{suffix}"))


INDEX_KEY = "__IDX"


def load_data(path: Path, data, pbar, suffix=".jsonl"):
    paths = list(path.glob(f"*{suffix}"))
    paths = sorted(paths, key=lambda x: int(x.stem))

    for p in paths:
        tmp = []
        i = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                for k in d:
                    if ("input" in k or "output" in k) and isinstance(d[k], str):
                        d[k] = d[k].replace("<|imgpad|>", ".").replace("<|image_pad|>", ".")
                d[INDEX_KEY] = i
                tmp.append(d)
                i += 1
        data.append({"results": tmp})
        pbar.advance(1)
    return data


def center_word_with_equals_exactly(word: str, total_length: int, char: str = "=") -> str:
    if len(word) > total_length:
        return word

    padding = total_length - len(word)
    left_pad = (padding) // 2
    right_pad = (padding + 1) // 2
    return char * left_pad + " " + word + " " + char * right_pad


def highlight_keyword(content: str, keyword: Optional[str]):
    if not keyword:
        return Text(content)
    text = Text()
    parts = content.split(keyword)
    for i, part in enumerate(parts):
        text.append(part, style=None)
        if i < len(parts) - 1:
            # text.append(keyword, style=Style(color="#d154d1", bgcolor="yellow", bold=True))
            text.append(keyword, style="on #8f51b5")
    return text


help_doc = """
⌨️   keybinds：

- `f/esc`: find/cancel
- `tab/←/→`: change focus
- `j/k`: page down/up
- `g/G`: scroll home/end
- `n/N`: next example/step
- `p/P`: previous example/step
- `s`: switch display mode
  - plain text
  - rich table

"""


class JsonLineViewer(App):
    BINDINGS = [
        ("left", "focus_previous", "Focus Previous"),
        ("right", "focus_next", "Focus Next"),
        ("s", "swith_render", "switch render"),
        # control
        ("n", "next_sample", "Next Sample"),
        ("N", "next_step", "Next Step"),
        ("p", "previous_sample", "Previous Sample"),
        ("P", "previous_step", "Previous Step"),
        # search
        ("f", "toggle_search", "find"),
        ("enter", "next_search", "find next"),
        ("escape", "cancel_search", "cancel find"),
        # scroll
        ("j", "page_down", "page down"),
        ("k", "page_up", "page up"),
        ("g", "page_home", "page home"),
        ("G", "page_end", "page end"),
    ]

    CSS = """

    Select:focus > SelectCurrent {
        border: tall #8f51b5;
    }
    Select.-expanded > SelectCurrent {
        border: tall #8f51b5;
    }
    #select-container {
        width: 15%;
        height: 100%;
        align: center top;
    }
    #search-container {
        height: 10%;
        align: center top;
    }
    #search-box {
        width: 80%;
    }
    """

    def __init__(self, row_num: int, data: list, pbar):
        super().__init__()
        self.row_num = row_num  # step num
        self.result_num: Optional[int] = None
        self.data = data
        self.render_table = False
        self.selected_row_index = 0
        self.selected_result_index = 0
        self.pbar = pbar

        self.matches = []
        self.current_match_index = 0

        self.highlighter = Highlighter()

        self.filter_fields = [(f, f, True) for f in data[0]["results"][0].keys()]

    def compose(self) -> ComposeResult:
        with Horizontal(id="search-container"):
            yield Input(placeholder="find something...", id="search-box")
            with Vertical(id="search-container2"):
                yield self.pbar
                yield Static("", id="search-status")

        with Horizontal():
            with Vertical(id="select-container"):
                yield Static("\n")
                yield Static(
                    renderable=Markdown(
                        help_doc,
                    ),
                    markup=False,
                )
                yield Static("\n")
                yield Select(
                    id="row-select",
                    value=0,
                    prompt="select step",
                    options=[("step: 0", 0)],
                    allow_blank=False,
                )
                yield Select(
                    id="result-select",
                    value=0,
                    prompt="select sample",
                    options=[("sample: 0", 0)],
                    allow_blank=False,
                )
                yield Select(
                    id="result-sort",
                    value=0,
                    prompt="排序",
                    options=[
                        ("sort", 0),
                        ("score asc", 1),
                        ("score desc", 2),
                    ],
                    allow_blank=False,
                )

                yield SelectionList[int](("Select ALL", 1, True), id="fields-select-all")
                with VerticalScroll(id="scroll-view2"):
                    yield SelectionList[str](*self.filter_fields, id="fields-select")
            with VerticalScroll(id="scroll-view"):
                yield Static(id="content", markup=False)

    def on_mount(self) -> None:
        self.row_select = self.query_one("#row-select", Select)
        self.result_select = self.query_one("#result-select", Select)
        self.result_sort = self.query_one("#result-sort", Select)
        self.content_display = self.query_one("#content", Static)
        self.search_box = self.query_one("#search-box", Input)
        self.scroll_view = self.query_one("#scroll-view", VerticalScroll)
        self.search_status = self.query_one("#search-status", Static)
        self.fields_select = self.query_one("#fields-select", SelectionList)
        self.fields_select.border_title = "field filter"

        if self.data:
            self.row_select.set_options([(f"step: {i}", i) for i in range(self.row_num)])
            # self.row_select.value = self.selected_row_index
            self.update_result_options()
            self.update_content()
            self.row_select.focus()

    def update_result_options(self, offset: int = 0, sort_desc: Optional[bool] = None):
        options = []
        if isinstance(self.selected_row_index, int) and self.selected_row_index < len(self.data):
            if self.result_num is None or sort_desc is not None:
                results = self.data[self.selected_row_index].get("results", [])
                if sort_desc is not None:
                    results = sorted(results, key=lambda x: x.get("score", x.get("score_1", 0)), reverse=sort_desc)

                options = [(f"sample: {r[INDEX_KEY]}", r[INDEX_KEY]) for r in results]
                self.result_select.set_options(options)
                self.result_num = len(results)

            if sort_desc is not None and options:
                self.selected_result_index = options[0][1]
            else:
                self.selected_result_index = offset

    def update_content(self, search_keyword: Optional[str] = None):
        content = ""
        try:
            results = self.data[self.selected_row_index].get("results", [])
            content_dict = results[self.selected_result_index]
            content_dict = {k: v for k, v in content_dict.items() if k in self.fields_select.selected}
            if self.render_table:
                content = Table("key", "value", show_lines=True)
                for k in content_dict:
                    v = content_dict[k]
                    v = json.dumps(v, ensure_ascii=False, indent=4) if not isinstance(v, str) else v
                    content.add_row(
                        k,
                        self.highlighter(highlight_keyword(v, search_keyword)),
                    )
            else:
                text = Text()
                for k in content_dict:
                    v = content_dict[k]
                    v = json.dumps(v, ensure_ascii=False, indent=4) if not isinstance(v, str) else v

                    s = center_word_with_equals_exactly(k, 64) + "\n"
                    s += v
                    s += "\n"
                    text.append(highlight_keyword(s, search_keyword))
                content = self.highlighter(text)
        except IndexError:
            content = f"Loading data async，progress: {len(self.data)}/{self.row_num} step"
        except Exception:
            content = self.highlighter(traceback.format_exc())

        self.content_display.update(content)

    @on(Select.Changed, "#row-select")
    def row_changed(self, event):
        self.selected_row_index = event.value
        self.update_result_options()
        self.update_content()

    @on(Select.Changed, "#result-select")
    def result_changed(self, event):
        self.selected_result_index = event.value
        self._clear_search()
        self.update_content()

    @on(Select.Changed, "#result-sort")
    def sort_changed(self, event):
        v = event.value
        self.update_result_options(sort_desc=None if v == 0 else False if v == 1 else True)
        self.update_content()

    @on(SelectionList.SelectedChanged, "#fields-select")
    def fields_changed(self, event):
        self.update_content()

    @on(SelectionList.SelectedChanged, "#fields-select-all")
    def fields_all_changed(self, event):
        s = self.query_one("#fields-select-all", SelectionList)
        if s.selected:
            self.fields_select.select_all()
        else:
            self.fields_select.deselect_all()

    def action_focus_previous(self):
        self.screen.focus_previous()

    def action_focus_next(self):
        self.screen.focus_next()

    def action_next_step(self) -> None:
        self.selected_row_index += 1
        if self.selected_row_index >= self.row_num:
            self.selected_row_index = 0
        self.row_select.value = self.selected_row_index
        self.update_result_options()
        self.update_content()

    def action_next_sample(self) -> None:
        self.selected_result_index += 1
        if not self.result_num or self.selected_result_index >= self.result_num:
            self.selected_result_index = 0
        self.result_select.value = self.selected_result_index
        self._clear_search()
        self.update_content()

    def action_previous_step(self) -> None:
        self.selected_row_index -= 1
        if self.selected_row_index < 0:
            self.selected_row_index = self.row_num - 1
        self.row_select.value = self.selected_row_index
        self.update_result_options()
        self.update_content()

    def action_previous_sample(self) -> None:
        self.selected_result_index -= 1
        if self.selected_result_index < 0:
            self.selected_result_index = self.result_num - 1
        self.result_select.value = self.selected_result_index
        self._clear_search()
        self.update_content()

    def action_swith_render(self):
        self.render_table = not self.render_table
        self.update_content()

    def action_toggle_search(self) -> None:
        self.search_box.focus()

    def action_cancel_search(self) -> None:
        self.search_box.value = ""
        self._clear_search()
        self.update_content()

    def _clear_search(self):
        self.matches = []
        self.search_status.update("")
        self.current_match_index = 0

    @on(Input.Submitted, "#search-box")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        self.matches = []
        self.current_match_index = 0
        if event.value:
            self.update_content(event.value)
            renderable = self.content_display.render()
            if isinstance(renderable, Table):
                return

            assert isinstance(renderable, Text), f"Expected Text, got {type(renderable)}"
            console = self.content_display._console
            lines = renderable.wrap(console, self.scroll_view.container_size.width)
            line_idx_recorded = set()
            for line_idx, line in enumerate(lines):
                if line_idx in line_idx_recorded:
                    continue
                if event.value in line:
                    self.matches.append(
                        {
                            "line": line_idx,
                            "word": event.value,
                        }
                    )
                    line_idx_recorded.add(line_idx)
            self.scroll_view.focus()
            self.action_next_search()

    def action_next_search(self) -> None:
        if not self.matches or self.current_match_index >= len(self.matches):
            return

        target_line = self.matches[self.current_match_index]["line"]
        self.scroll_view.scroll_to(x=0, y=target_line * 1, animate=False)
        self.current_match_index = (self.current_match_index + 1) % len(self.matches)
        self.search_status.update(
            Text(
                f"Find ：{self.current_match_index + 1}/{len(self.matches)}",
                style="bold on #8f51b5",
            )
        )

    def action_page_up(self):
        self.scroll_view.scroll_page_up(animate=False)

    def action_page_down(self):
        self.scroll_view.scroll_page_down(animate=False)

    def action_page_home(self):
        self.scroll_view.scroll_home(animate=False)

    def action_page_end(self):
        self.scroll_view.scroll_end(animate=False)


app = typer.Typer()


@app.command()
def run(path: Path):
    path = Path(path)
    assert path.exists(), f"{path} does not exist"
    file_num = get_jsonl_file_num(path)
    pbar = ProgressBar(total=file_num, name="data loading")
    data = []
    load_data_partial = partial(load_data, path, data, pbar)
    t = threading.Thread(target=load_data_partial, daemon=True)
    t.start()
    while not data:
        time.sleep(1)
    app = JsonLineViewer(row_num=file_num, data=data, pbar=pbar)
    app.run()


if __name__ == "__main__":
    app()
