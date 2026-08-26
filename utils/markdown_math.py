r"""LaTeX 公式支持：markdown-it 解析规则 + KaTeX 前端渲染资源。

上游报告里的公式用 \( ... \) / \[ ... \] 包裹（偶尔是 $$ ... $$ 或裸的 \begin{...} 环境）。
直接交给 CommonMark 渲染会被行内规则啃掉——\[ 变成 [、\\ 变成 \、X_d...Y_d 被当成斜体，
于是 "\[ \tau_d=\begin{bmatrix}X_d\\Y_d\\N_d\end{bmatrix} \]" 会显示成
"[ \tau_d= \begin{bmatrix} X_d\Y_d\N_d \end{bmatrix}" 这种乱码。

这里在解析阶段就把整段公式原样抓出来（不参与任何行内规则），输出为
<div class="math-block"> / <span class="math-inline">，再由浏览器端 KaTeX 渲染。
CDN 不可达时元素里留下的是可读的 TeX 源码，属于降级而非更差。
"""

from __future__ import annotations

import html
import re
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from markdown_it.rules_inline import StateInline

# 行内/块级成对定界符；\begin{env} ... \end{env} 单独处理（内容需连同定界符一起交给 KaTeX）
_INLINE_PAIRS = (("\\[", "\\]", True), ("\\(", "\\)", False))
_BLOCK_PAIRS = (("\\[", "\\]"), ("$$", "$$"))
_BEGIN_RE = re.compile(r"\\begin\{([A-Za-z]+\*?)\}")
_TEX_CMD_RE = re.compile(r"\\[A-Za-z]")
_SHELL_VAR_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_MATH_CHAR_RE = re.compile(r"[\^_{}]")
_MAX_DOLLAR_SPAN = 200


# --------------------------------------------------------------------------- 块级


def _line_text(state: StateBlock, line: int) -> str:
    return state.src[state.bMarks[line] + state.tShift[line] : state.eMarks[line]]


def _math_block(state: StateBlock, start_line: int, end_line: int, silent: bool) -> bool:
    """整行以 \\[ / $$ / \\begin{env} 开头的公式块（允许跨行、跨空行）。"""
    if state.sCount[start_line] - state.blkIndent >= 4:
        return False  # 四空格缩进属于代码块
    first = _line_text(state, start_line)
    keep = False
    for open_d, close_d in _BLOCK_PAIRS:
        if first.startswith(open_d):
            opener, closer = open_d, close_d
            break
    else:
        match = _BEGIN_RE.match(first)
        if match is None:
            return False
        opener, closer, keep = match.group(0), "\\end{%s}" % match.group(1), True

    parts: list[str] = []
    last_line = -1
    rest = first[len(opener) :]
    hit = rest.find(closer)
    if hit >= 0:
        if rest[hit + len(closer) :].strip():
            return False  # 闭合标记后还有正文 → 交给行内规则，避免吞掉后半句
        parts.append(rest[:hit])
        last_line = start_line
    else:
        parts.append(rest)
        line = start_line + 1
        while line < end_line:
            text = _line_text(state, line)
            hit = text.find(closer)
            if hit >= 0:
                if text[hit + len(closer) :].strip():
                    return False
                parts.append(text[:hit])
                last_line = line
                break
            parts.append(text)
            line += 1
        if last_line < 0:
            return False  # 没有闭合标记 → 按普通段落处理

    body = "\n".join(parts).strip()
    if not body:
        return False
    if silent:
        return True
    token = state.push("math_block", "", 0)
    token.block = True
    token.markup = opener
    token.content = f"{opener}{body}{closer}" if keep else body
    token.map = [start_line, last_line + 1]
    state.line = last_line + 1
    return True


# --------------------------------------------------------------------------- 行内


def _push(state: StateInline, silent: bool, tex: str, display: bool, new_pos: int) -> bool:
    if not silent:
        token = state.push("math_inline", "", 0)
        token.content = tex
        token.meta = {"display": display}
    state.pos = new_pos
    return True


def _scan_pair(state: StateInline, silent: bool, open_d: str, close_d: str,
               display: bool, *, keep: bool) -> bool:
    src = state.src
    start = state.pos + len(open_d)
    hit = src.find(close_d, start)
    if hit < 0:
        return False
    body = src[start:hit].strip()
    if not body:
        return False
    tex = f"{open_d}{body}{close_d}" if keep else body
    return _push(state, silent, tex, display, hit + len(close_d))


def _looks_like_math(body: str) -> bool:
    """$...$ 的保守判定：报告里的 $ 绝大多数是价格或 shell 变量，误判会整段吞掉正文。"""
    if "\n\n" in body:
        return False
    if _TEX_CMD_RE.search(body):
        return True  # 含 \frac、\alpha 等控制序列，基本可确定是公式
    if len(body) == 1 and body.isalpha():
        return True  # $x$
    if _SHELL_VAR_RE.match(body):
        return False  # $DATASET_PATH、$HOME 这类变量名
    return bool(_MATH_CHAR_RE.search(body)) and " " not in body and len(body) <= 32


def _scan_dollar(state: StateInline, silent: bool) -> bool:
    src = state.src
    start = state.pos + 1
    if start >= len(src) or src[start].isspace():
        return False  # "$ 5" 之类，开定界符后不能紧跟空白
    hit = src.find("$", start)
    while hit > 0 and src[hit - 1] == "\\":  # 跳过公式内的 \$
        hit = src.find("$", hit + 1)
    if hit < 0 or hit - start > _MAX_DOLLAR_SPAN:
        return False
    if src[hit - 1].isspace() or src[hit + 1 : hit + 2].isdigit():
        return False  # "约 $0.6/h，批量 $0.3/h" 这类价格
    body = src[start:hit]
    if not _looks_like_math(body):
        return False
    return _push(state, silent, body, False, hit + 1)


def _math_inline(state: StateInline, silent: bool) -> bool:
    src, pos = state.src, state.pos
    char = src[pos]
    if char == "$":
        if src.startswith("$$", pos):
            return _scan_pair(state, silent, "$$", "$$", True, keep=False)
        return _scan_dollar(state, silent)
    if char != "\\":
        return False
    for open_d, close_d, display in _INLINE_PAIRS:
        if src.startswith(open_d, pos):
            return _scan_pair(state, silent, open_d, close_d, display, keep=False)
    match = _BEGIN_RE.match(src, pos)
    if match is None:
        return False
    return _scan_pair(state, silent, match.group(0), "\\end{%s}" % match.group(1), True, keep=True)


# --------------------------------------------------------------------------- 渲染


def _escape_tex(text: str) -> str:
    return html.escape(text, quote=False)


def _render_math_inline(tokens: list[Any], idx: int, options: Any, env: Any) -> str:
    token = tokens[idx]
    cls = "math-block" if (token.meta or {}).get("display") else "math-inline"
    return f'<span class="{cls}">{_escape_tex(token.content)}</span>'


def _render_math_block(tokens: list[Any], idx: int, options: Any, env: Any) -> str:
    return f'<div class="math-block">{_escape_tex(tokens[idx].content)}</div>\n'


def install_math(md: MarkdownIt) -> MarkdownIt:
    """给 markdown-it 实例装上公式规则（须在 escape 规则之前拦截反斜杠）。"""
    md.inline.ruler.before("escape", "math_inline", _math_inline)
    md.block.ruler.before(
        "fence", "math_block", _math_block,
        {"alt": ["paragraph", "reference", "blockquote", "list"]},
    )
    md.renderer.rules["math_inline"] = _render_math_inline
    md.renderer.rules["math_block"] = _render_math_block
    return md


def has_math(rendered_html: str) -> bool:
    """页面是否需要加载 KaTeX。"""
    return 'class="math-block"' in rendered_html or 'class="math-inline"' in rendered_html


# --------------------------------------------------------------------------- 前端资源

_KATEX_CDN = "https://cdn.jsdelivr.net/npm/katex@0.16"

# <head> 里的样式表（KaTeX 字体随 CSS 一起从 CDN 拉取）
KATEX_HEAD = f'<link rel="stylesheet" href="{_KATEX_CDN}/dist/katex.min.css">'

MATH_CSS = """
.math-block{display:block;margin:14px 0;padding:2px 0;text-align:center;overflow-x:auto;overflow-y:hidden}
.math-inline{overflow-wrap:normal;word-break:keep-all}
.katex{font-size:1.05em;overflow-wrap:normal;word-break:keep-all}
.katex-display{margin:0}
.math-raw{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em}
"""

# 放在 </body> 前：katex.min.js 先按顺序加载完，再由内联脚本逐个渲染
KATEX_JS = f"""<script src="{_KATEX_CDN}/dist/katex.min.js"></script>
<script>
(function(){{
  var nodes = document.querySelectorAll('.math-block,.math-inline');
  if(!nodes.length) return;
  if(!window.katex){{ console.error('katex not loaded'); return; }}
  nodes.forEach(function(el){{
    var tex = el.textContent;
    try{{
      katex.render(tex, el, {{
        displayMode: el.classList.contains('math-block'),
        throwOnError: false, strict: 'ignore', trust: false
      }});
    }}catch(err){{
      el.classList.add('math-raw'); el.textContent = tex;
      console.error('katex', err);
    }}
  }});
}})();
</script>"""
