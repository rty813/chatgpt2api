"""报告 markdown 里的 LaTeX 公式：解析阶段原样保留，输出交给 KaTeX 渲染。"""
import unittest

from markdown_it import MarkdownIt

from utils.markdown_math import has_math, install_math

MD = install_math(
    MarkdownIt("commonmark", {"html": False, "linkify": True}).enable("table").enable("strikethrough")
)


class MathBlockTests(unittest.TestCase):
    def test_display_block_keeps_backslashes(self) -> None:
        # 未处理时 CommonMark 会渲染成 "[ \tau_d= \begin{bmatrix} X_d\Y_d\N_d \end{bmatrix}"
        out = MD.render("则期望广义力为\n\n\\[\n\\tau_d=\n\\begin{bmatrix}\nX_d\\\\Y_d\\\\N_d\n\\end{bmatrix}\n\\]\n\n其中……")
        self.assertIn('<div class="math-block">', out)
        self.assertIn("\\tau_d=", out)
        self.assertIn("\\begin{bmatrix}", out)
        self.assertIn("X_d\\\\Y_d\\\\N_d", out)  # 换行符 \\ 没有被吃成 \
        self.assertNotIn("<em>", out)  # 下标 _d 没有被当成斜体
        self.assertTrue(has_math(out))

    def test_dollar_dollar_block(self) -> None:
        out = MD.render("$$\nE=mc^2\n$$")
        self.assertIn('<div class="math-block">E=mc^2</div>', out)

    def test_bare_environment_keeps_delimiters(self) -> None:
        out = MD.render("\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}")
        self.assertIn('<div class="math-block">\\begin{aligned}', out)
        self.assertIn("\\end{aligned}</div>", out)

    def test_block_spanning_blank_line(self) -> None:
        out = MD.render("\\[\nA=1\n\n B=2\n\\]")
        self.assertIn('<div class="math-block">', out)
        self.assertIn("B=2", out)

    def test_trailing_text_falls_back_to_inline(self) -> None:
        out = MD.render("\\[x=1\\] 就是解。")
        self.assertIn('<span class="math-block">x=1</span>', out)
        self.assertIn("就是解。", out)

    def test_unclosed_block_is_plain_paragraph(self) -> None:
        out = MD.render("\\[\nx=1\n")
        self.assertNotIn("math-block", out)

    def test_indented_code_is_not_math(self) -> None:
        out = MD.render("    \\[x=1\\]\n")
        self.assertIn("<pre><code>", out)
        self.assertNotIn("math-block", out)


class MathInlineTests(unittest.TestCase):
    def test_paren_delimiters(self) -> None:
        out = MD.render("位于 \\((x_i,y_i)\\) 的第 \\(i\\) 个推进器")
        self.assertIn('<span class="math-inline">(x_i,y_i)</span>', out)
        self.assertIn('<span class="math-inline">i</span>', out)
        self.assertNotIn("<em>", out)

    def test_inline_inside_table_cell(self) -> None:
        out = MD.render("| 符号 | 含义 |\n| --- | --- |\n| \\(\\alpha_i\\) | 方位角 |")
        self.assertIn('<span class="math-inline">\\alpha_i</span>', out)

    def test_escapes_html_in_tex(self) -> None:
        out = MD.render("\\(a < b \\& c\\)")  # TeX 源码原样保留，只做 HTML 转义
        self.assertIn("a &lt; b \\&amp; c", out)
        self.assertNotIn("<b", out)

    def test_code_span_and_fence_untouched(self) -> None:
        out = MD.render("行内 `\\(x\\)` 与\n\n```text\n\\[y=1\\]\n```")
        self.assertNotIn("math-inline", out)
        self.assertNotIn("math-block", out)

    def test_backslash_escape_still_works(self) -> None:
        out = MD.render("\\*不是斜体\\* 与 \\[]")
        self.assertNotIn("math", out)
        self.assertIn("*不是斜体*", out)


class DollarHeuristicTests(unittest.TestCase):
    """$ 在报告里绝大多数是价格/变量名，误判会整段吞掉正文。"""

    def test_prices_are_not_math(self) -> None:
        for text in ("约 $0.60/hr streaming，$0.36/hr batch（示例区）",
                     "公开零售案例约 AU$675；具体多按尺寸询价。",
                     "赎金 $5,000 到 $10,000 不等"):
            with self.subTest(text=text):
                self.assertNotIn("math-inline", MD.render(text))

    def test_shell_variable_is_not_math(self) -> None:
        self.assertNotIn("math-inline", MD.render("把 $DATASET_PATH 换成 $IMAGE_PATH 即可"))

    def test_tex_command_is_math(self) -> None:
        self.assertIn('<span class="math-inline">\\alpha_i</span>', MD.render("夹角 $\\alpha_i$ 为"))

    def test_single_letter_is_math(self) -> None:
        self.assertIn('<span class="math-inline">x</span>', MD.render("变量 $x$ 表示位移"))


class PageIntegrationTests(unittest.TestCase):
    def test_katex_injected_only_when_math_present(self) -> None:
        from api.deep_research import _render_status_html

        base = {"id": "a" * 32, "status": "success", "title": "船舶 Joystick", "sources_count": 3}
        with_math = _render_status_html({**base, "report_markdown": "# 报告\n\n\\[\nX=1\n\\]"})
        self.assertIn("katex.min.css", with_math)
        self.assertIn("katex.min.js", with_math)
        self.assertIn('<div class="math-block">X=1</div>', with_math)

        without_math = _render_status_html({**base, "report_markdown": "# 报告\n\n没有公式。"})
        self.assertNotIn("katex.min.css", without_math)  # 无公式时不拉 CDN（内联样式表除外）
        self.assertNotIn("katex.min.js", without_math)


if __name__ == "__main__":
    unittest.main()
