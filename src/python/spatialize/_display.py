"""How Spatialize shows its progress and its messages: one look in a terminal, in an IDE and in a
Jupyter notebook, with the colours of the ``alges`` palette.

The session setting ``display`` chooses the renderer (:mod:`spatialize.session`). With ``"auto"``,
the default, :func:`environment` detects it: a Jupyter kernel (JupyterLab, the classic notebook,
VS Code and PyCharm notebooks, Colab) gets HTML; a terminal, including the terminals of the IDEs and
PyCharm's run console, gets rich's live bars; any other output (a file, a pipe, a continuous
integration log) gets plain lines.
"""
import html
import os
import sys
import time

# the alges palette of spatialize.viz, with an accent for each level
COLOURS = {"brand": "#4b87af", "brand_dark": "#326c94", "brand_light": "#a7c9e1", "success": "#59a590",
           "debug": "#8a94a0", "info": "#4b87af", "warning": "#d9962b", "error": "#c9544a", "critical": "#a32d2d"}
SYMBOLS = {"debug": "·", "info": "ℹ", "warning": "▲", "error": "✖", "critical": "✖"}
MODES = ("auto", "terminal", "notebook", "plain", "silent")


def environment():
    """The kind of output Spatialize writes to: ``"notebook"``, ``"terminal"`` or ``"plain"``."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            name = type(shell).__name__
            if name == "ZMQInteractiveShell" or "google.colab" in type(shell).__module__ \
                    or "IPKernelApp" in getattr(shell, "config", {}):
                return "notebook"
    except Exception:
        pass
    if os.environ.get("TERM") == "dumb":
        return "plain"
    if os.environ.get("PYCHARM_HOSTED") or os.environ.get("FORCE_COLOR"):
        return "terminal"
    try:
        if sys.stderr.isatty():
            return "terminal"
    except (AttributeError, ValueError):
        pass
    return "plain"


def mode():
    """The renderer in effect: the session's ``display``, ``"auto"`` resolved by
    :func:`environment`."""
    from spatialize import session
    chosen = session.get("display")
    return environment() if chosen == "auto" else chosen


def _elapsed(seconds):
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else (f"{m}m {s:02d}s" if m else f"{seconds}s")


def _precise_elapsed(seconds):
    return f"{seconds:.1f}s" if seconds < 60 else _elapsed(seconds)


# ---------------------------------------------------------------------------------------------- terminal
_console = None


def _theme():
    from rich.theme import Theme
    c = COLOURS
    return Theme({"progress.percentage": f"bold {c['brand_dark']}", "progress.elapsed": "dim",
                  "progress.remaining": "dim", "progress.download": c["brand_dark"],
                  "progress.spinner": c["brand"], "table.title": f"bold {c['brand']}"})


def _rich_console():
    global _console
    if _console is None:
        from rich.console import Console
        _console = Console(stderr=True, highlight=False, soft_wrap=False, theme=_theme())
    return _console


_ENGINE = None


def tidy(text):
    """A message of the compiled engine, ``"[C++|mondrian/idw] computing estimates"``, as
    ``"computing estimates · mondrian/idw"``; other text unchanged."""
    global _ENGINE
    if _ENGINE is None:
        import re
        _ENGINE = re.compile(r"^\[C\+\+\|(?P<where>[^\]]+)\]\s*(?P<what>.+)$")
    hit = _ENGINE.match(str(text))
    return f"{hit.group('what').strip()} · {hit.group('where')}" if hit else str(text)


class _TerminalProgress:
    def __init__(self, total, desc):
        from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
                                   TimeElapsedColumn, TimeRemainingColumn)
        self.total, self.desc, self.start = total, desc, time.perf_counter()
        c = COLOURS
        self.progress = Progress(
            SpinnerColumn(style=c["brand"]),
            TextColumn(f"[bold {c['brand']}]spatialize[/] [dim]·[/] {{task.description}}"),
            BarColumn(bar_width=32, style="grey35", complete_style=c["brand"],
                      finished_style=c["success"], pulse_style=c["brand"]),
            TaskProgressColumn(),
            TextColumn(f"[{c['brand_dark']}]{{task.completed}}/{{task.total}}[/]"),
            TextColumn("[dim]•[/]"), TimeElapsedColumn(),
            TextColumn("[dim]<[/]"), TimeRemainingColumn(),
            console=_rich_console(), transient=True, refresh_per_second=8)
        from rich.markup import escape
        self.progress.start()
        self.desc = escape(desc)
        self.task = self.progress.add_task(self.desc, total=total)

    def advance(self, n):
        self.progress.advance(self.task, n)

    def finish(self):
        self.progress.stop()
        c = COLOURS
        _rich_console().print(
            f"[{c['success']}]✔[/] [bold {c['brand']}]spatialize[/] [dim]·[/] {self.desc} "
            f"[dim]· {self.total} in {_precise_elapsed(time.perf_counter() - self.start)}[/]")


def _terminal_message(level, text):
    from rich.text import Text
    c = COLOURS
    line = Text()
    line.append(f"{SYMBOLS[level]} ", style=c[level])
    line.append("spatialize", style=f"bold {c['brand']}")
    line.append(" · ", style="dim")
    line.append(f"{level.upper():8s}", style=f"bold {c[level]}")
    line.append(text, style="dim" if level == "debug" else "")
    _rich_console().print(line)


# ---------------------------------------------------------------------------------------------- notebook
_CSS = """<style>
.sptlz{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:12.5px;
 line-height:1.45;margin:3px 0;display:flex;align-items:center;gap:8px;color:inherit}
.sptlz .brand{font-weight:600;color:%(brand)s;letter-spacing:.2px}
.sptlz .dim{opacity:.6}
.sptlz .badge{font-size:10.5px;font-weight:700;letter-spacing:.4px;padding:1px 7px;border-radius:9px;color:#fff}
.sptlz .track{flex:0 0 240px;height:8px;border-radius:4px;background:rgba(75,135,175,.18);overflow:hidden}
.sptlz .fill{height:100%%;border-radius:4px;background:linear-gradient(90deg,%(brand_dark)s,%(brand)s);transition:width .25s}
.sptlz .fill.done{background:%(success)s}
.sptlz .num{font-variant-numeric:tabular-nums}
.sptlz .icon{display:inline-block;width:1.1em;text-align:center}
.sptlz .desc{display:inline-block;min-width:330px}
.sptlz table{font-size:12.5px;border-collapse:collapse;margin:4px 0 0 1.6em}
.sptlz td{padding:2px 16px 2px 0;text-align:left}
</style>""" % COLOURS
_css_shown = False


def _html(body):
    global _css_shown
    from IPython.display import HTML
    out = HTML(("" if _css_shown else _CSS) + body)
    _css_shown = True
    return out


class _NotebookProgress:
    def __init__(self, total, desc):
        from IPython.display import display
        self.total, self.desc, self.count = total, html.escape(desc), 0
        self.start = self.last = time.perf_counter()
        self.handle = display(_html(self._body()), display_id=True)

    def _body(self, done=False):
        frac = 1.0 if done else (self.count / self.total if self.total else 0.0)
        elapsed = time.perf_counter() - self.start
        if done:
            tail = f'<span class="dim num">{self.total} in {_precise_elapsed(elapsed)}</span>'
            icon = f'<span class="icon" style="color:{COLOURS["success"]}">✔</span>'
        else:
            eta = elapsed * (1 - frac) / frac if frac > 0 else 0.0
            tail = (f'<span class="num">{100 * frac:3.0f}%</span>'
                    f'<span class="dim num">{self.count}/{self.total} · {_elapsed(elapsed)} &lt; {_elapsed(eta)}</span>')
            icon = f'<span class="icon" style="color:{COLOURS["brand"]}">◌</span>'
        return (f'<div class="sptlz">{icon}<span class="brand">spatialize</span><span class="dim">·</span>'
                f'<span class="desc">{self.desc}</span><div class="track"><div class="fill{" done" if done else ""}" '
                f'style="width:{100 * frac:.1f}%"></div></div>{tail}</div>')

    def advance(self, n):
        self.count = min(self.total, self.count + n)
        now = time.perf_counter()
        if now - self.last >= 0.1 or self.count >= self.total:
            self.last = now
            self.handle.update(_html(self._body()))

    def finish(self):
        self.handle.update(_html(self._body(done=True)))


def _notebook_message(level, text):
    from IPython.display import display
    colour = COLOURS[level]
    display(_html(
        f'<div class="sptlz"><span class="icon" style="color:{colour}">{SYMBOLS[level]}</span>'
        f'<span class="brand">spatialize</span><span class="dim">·</span>'
        f'<span class="badge" style="background:{colour}">{level.upper()}</span>'
        f'<span{" class=dim" if level == "debug" else ""}>{html.escape(text)}</span></div>'))


# ---------------------------------------------------------------------------------------------- plain
class _PlainProgress:
    def __init__(self, total, desc):
        self.total, self.desc, self.count, self.next = total, desc, 0, 25
        self.start = time.perf_counter()
        print(f"spatialize | {desc}: started, {total} steps", file=sys.stderr, flush=True)

    def advance(self, n):
        self.count = min(self.total, self.count + n)
        pct = 100 * self.count // self.total if self.total else 100
        while pct >= self.next and self.next < 100:
            print(f"spatialize | {self.desc}: {self.next}% ({self.count}/{self.total}, "
                  f"{_elapsed(time.perf_counter() - self.start)})", file=sys.stderr, flush=True)
            self.next += 25

    def finish(self):
        print(f"spatialize | {self.desc}: done, {self.total} in {_precise_elapsed(time.perf_counter() - self.start)}",
              file=sys.stderr, flush=True)


def _plain_message(level, text):
    print(f"spatialize | {level.upper()}: {text}", file=sys.stderr, flush=True)


class _Silent:
    def __init__(self, total, desc):
        pass

    def advance(self, n):
        pass

    def finish(self):
        pass


_PROGRESS = {"terminal": _TerminalProgress, "notebook": _NotebookProgress, "plain": _PlainProgress,
             "silent": _Silent}
_MESSAGE = {"terminal": _terminal_message, "notebook": _notebook_message, "plain": _plain_message,
            "silent": lambda level, text: None}


def progress(total, desc):
    """A progress display of ``total`` steps, with ``advance(n)`` and ``finish()``. A renderer that
    fails (a notebook without IPython's display, say) falls back to plain lines."""
    m = mode()
    try:
        return _PROGRESS[m](total, desc)
    except Exception:
        return _PlainProgress(total, desc)


def message(level, text):
    """Show one message of ``level`` (``"debug"``, ``"info"``, ``"warning"``, ``"error"`` or
    ``"critical"``)."""
    text = tidy(text)
    m = mode()
    try:
        _MESSAGE[m](level, text)
    except Exception:
        _plain_message(level, text)


def settings_table(rows):
    """Show the session settings, ``rows`` of ``(name, value, changed)``, the changed ones marked."""
    m = mode()
    if m == "silent":
        return
    try:
        if m == "terminal":
            from rich.table import Table
            from rich import box
            table = Table(title=f"[bold {COLOURS['brand']}]spatialize[/] [dim]· session[/]", title_justify="left",
                          title_style="",
                          box=box.SIMPLE_HEAD, header_style=f"bold {COLOURS['brand_dark']}", pad_edge=False)
            table.add_column("setting")
            table.add_column("value")
            table.add_column("")
            for name, value, changed in rows:
                table.add_row(name, repr(value), f"[{COLOURS['warning']}]set[/]" if changed else "[dim]default[/]")
            _rich_console().print(table)
            return
        if m == "notebook":
            from IPython.display import display
            body = "".join(
                f'<tr><td>{html.escape(name)}</td>'
                f'<td style="font-family:ui-monospace,Menlo,Consolas,monospace">{html.escape(repr(value))}</td>'
                f'<td>' + (f'<span class="badge" style="background:{COLOURS["warning"]}">SET</span>'
                                                   if changed else '<span class="dim">default</span>') + '</td></tr>'
                for name, value, changed in rows)
            display(_html(f'<div class="sptlz" style="display:block"><div><span class="icon"></span>'
                          f'<span class="brand">spatialize</span><span class="dim"> · session</span></div>'
                          f'<table>{body}</table></div>'))
            return
    except Exception:
        pass
    width = max(len(name) for name, _, _ in rows)
    print("spatialize | session", file=sys.stderr)
    for name, value, changed in rows:
        print(f"  {name:{width}s}  {value!r}{'   (set)' if changed else ''}", file=sys.stderr)
