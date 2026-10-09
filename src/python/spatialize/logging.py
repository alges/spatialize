import json
import logging
import time


from spatialize._util import SingletonType

# Spatialize's own logger: it leaves the logging of the application alone (no handler on the root
# logger) and shows its messages through spatialize._display, in the look of the session's
# ``display``. Its level is the session's ``verbosity`` unless set with ``log.setLevel``.
log = logging.getLogger("spatialize")
log.propagate = False

_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}


class _DisplayHandler(logging.Handler):
    def emit(self, record):
        from spatialize import _display
        name = record.levelname.lower()
        _display.message(name if name in _display.SYMBOLS else "info", record.getMessage())


if not any(isinstance(h, _DisplayHandler) for h in log.handlers):
    log.addHandler(_DisplayHandler())


def _threshold():
    """The lowest level shown: the one set on ``log`` by the user, otherwise the session's
    ``verbosity``."""
    if log.level != logging.NOTSET:
        return log.level
    from spatialize import session
    return _LEVELS[session.get("verbosity")]


# ************************************ PROTOCOL ************************************
# Messages for logging:
#
#   {"message": {"text": "<the log text>", "level": "<DEBUG|INFO|WARNING|ERROR|CRITICAL>"}}
#
# Messages for showing progress:
#
#  1. To start a new progress counting (``desc``, optional, names the task):
#   {"progress": {"init": <total expected count>, "step": <increment step>, "desc": <name>}}
#
#  2. To inform during a progress counting
#   {"progress": {"token": <value>}}
#
#  3. To finish a progress counting
#   {"progress": "done"}
# **********************************************************************************
class level:
    debug = "DEBUG"
    info = "INFO"
    warn = "WARNING"
    error = "ERROR"
    critical = "CRITICAL"


class logger:
    message = "message"
    text = "text"
    level = "level"

    @classmethod
    def debug(cls, msg):
        return {cls.message: {cls.text: msg, cls.level: level.debug}}

    @classmethod
    def info(cls, msg):
        return {cls.message: {cls.text: msg, cls.level: level.info}}

    @classmethod
    def warning(cls, msg):
        return {cls.message: {cls.text: msg, cls.level: level.warn}}

    @classmethod
    def error(cls, msg):
        return {cls.message: {cls.text: msg, cls.level: level.error}}

    @classmethod
    def critical(cls, msg):
        return {cls.message: {cls.text: msg, cls.level: level.critical}}


class progress:
    init_ = "init"
    step = "step"
    token = "token"
    done = "done"

    prog = "progress"

    @classmethod
    def init(cls, total, increment_step=1, desc=None):
        msg = {cls.prog: {cls.init_: total, cls.step: increment_step}}
        if desc:
            msg[cls.prog]["desc"] = desc
        return msg

    @classmethod
    def stop(cls):
        return {cls.prog: cls.done}

    @classmethod
    def inform(cls, value=None):
        return {cls.prog: {cls.token: value}}


# **************************++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ********************************* HANDLERS ***************************************
class MessageHandler:
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __call__(self, msg):
        for callback in self.callbacks:
            callback(msg)


class LogMessage:  # callback function
    """Shows the logging protocol's messages above the session's ``verbosity``. With
    ``engine_as_progress``, the compiled engine's announcements of its runs (``"[C++|...]
    computing ..."``) are left to the progress display, which names its runs after them."""

    def __init__(self, engine_as_progress=False):
        self.engine_as_progress = engine_as_progress

    def __call__(self, msg):
        global log

        try:
            m = self._pass_protocol(msg)
        except:
            return

        lev = m[logger.message][logger.level]
        text = m[logger.message][logger.text]
        if lev == level.info and str(text).startswith("[C++|") and self.engine_as_progress:
            return      # the progress display shows it as the name of its run
        number = logging.getLevelName(lev)
        if not isinstance(number, int) or number < _threshold():
            return
        log.handle(log.makeRecord(log.name, number, "spatialize", 0, text, None, None))

    @staticmethod
    def _pass_protocol(msg):
        m = msg
        if isinstance(msg, str):
            m = json.loads(msg)
        if isinstance(m, dict) and logger.message in m:
            return m
        raise TypeError


log_message = LogMessage()  # to use it in the plain python code


class AsyncProgressHandler:  # callback function

    def _done(self):
        self.elapsed_time = time.time() - self.start_time

    def __call__(self, msg):
        try:
            m = self._pass_protocol(msg)
        except:
            return

        if isinstance(m[progress.prog], dict):
            if progress.init_ in m[progress.prog]:
                self._init(m[progress.prog][progress.init_], m[progress.prog][progress.step])
                self._reset()
            if progress.token in m[progress.prog]:
                # no need to process the incoming value
                self._increment()
                self._update()
        if m[progress.prog] == progress.done:
            self._done()
            self._reset()

    def _init(self, total, step):
        self.start_time = time.time()
        self.total = total
        self.step = step

    def _reset(self):
        self.count = 0
        self.p = 0
        self.p_prev = -1
        self.ready_to_update = False

    def _increment(self):
        self.count += self.step

    @staticmethod
    def _pass_protocol(msg):
        m = msg
        if isinstance(msg, str):
            m = json.loads(msg)
        if isinstance(m, dict) and progress.prog in m:
            return m
        raise TypeError

    def _update(self):
        try:
            self.ready_to_update = False
            self.p = (self.count * 100) // self.total
            if self.p > self.p_prev:
                self.p_prev = self.p
                self.ready_to_update = True
        finally:
            pass


class DisplayProgress:  # callback function
    """Shows the progress protocol with spatialize._display, in the look of the session's
    ``display``. Progress runs may nest, a run started inside another being shown on its own. A run
    takes its name from the ``desc`` of its ``init`` message, otherwise from the last message of the
    compiled engine (``"[C++|mondrian/idw] computing estimates"`` gives "computing estimates ·
    mondrian/idw")."""

    def __init__(self):
        self.stack = []
        self.pending = None

    def __call__(self, msg):
        try:
            m = json.loads(msg) if isinstance(msg, str) else msg
        except (TypeError, ValueError):
            return
        if not isinstance(m, dict):
            return
        if logger.message in m:
            text = m[logger.message].get(logger.text, "") if isinstance(m[logger.message], dict) else ""
            if str(text).startswith("[C++|"):
                from spatialize import _display
                self.pending = _display.tidy(text)
            return
        if progress.prog not in m:
            return
        body = m[progress.prog]
        from spatialize import _display
        if isinstance(body, dict) and progress.init_ in body:
            desc = body.get("desc") or self.pending or "working"
            self.pending = None
            self.stack.append([_display.progress(int(body[progress.init_]), desc), int(body.get(progress.step, 1))])
        elif isinstance(body, dict) and progress.token in body:
            if self.stack:
                bar, step = self.stack[-1]
                bar.advance(step)
        elif body == progress.done and self.stack:
            self.stack.pop()[0].finish()


class SingletonDisplayProgress(DisplayProgress, metaclass=SingletonType):
    pass


# the names of earlier versions, now shown in the session's look
AsyncProgressCounter = DisplayProgress
AsyncProgressBar = DisplayProgress
SingletonAsyncProgressCounter = SingletonDisplayProgress


class SingletonNullMsgHandler(metaclass=SingletonType):
    def __call__(self, *args, **kwargs):
        pass


class SingletonMessageHandler(MessageHandler, metaclass=SingletonType):
    pass


class SingletonLogMessage(LogMessage, metaclass=SingletonType):
    def __init__(self):
        super().__init__(engine_as_progress=True)


# **************************++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def default_singleton_callback(msg):
    """The default callback: messages and progress shown in the look of the session's ``display``
    (:mod:`spatialize.session`)."""
    return SingletonMessageHandler([SingletonLogMessage(), SingletonDisplayProgress()])(msg)


def singleton_null_callback(msg):
    return SingletonNullMsgHandler()(msg)
