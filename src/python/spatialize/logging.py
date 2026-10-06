import json
import logging
import os
import time

import numpy as np
from rich.logging import RichHandler
from tqdm.auto import tqdm

from spatialize._util import SingletonType

FORMAT = "%(message)s"
logging.basicConfig(
    level="ERROR",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(show_time=False, show_path=False)]
)

log = logging.getLogger("rich")


# ************************************ PROTOCOL ************************************
# Messages for logging:
#
#   {"message": {"text": "<the log text>", "level": "<DEBUG|INFO|WARNING|ERROR|CRITICAL>"}}
#
# Messages for showing progress:
#
#  1. To start a new progress counting:
#   {"progress": {"init": <total expected count>, "step": <increment step>}}
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
    def init(cls, total, increment_step=1):
        return {cls.prog: {cls.init_: total, cls.step: increment_step}}

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

    def __init__(self):
        pass

    def __call__(self, msg):
        global log

        try:
            m = self._pass_protocol(msg)
        except:
            return

        lev = m[logger.message][logger.level]
        text = m[logger.message][logger.text]

        if lev == level.debug:
            log.debug(text)

        if lev == level.info:
            log.info(text)

        if lev == level.warn:
            log.warning(text)

        if lev == level.error:
            log.error(text)

        if lev == level.critical:
            log.critical(text)

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


class AsyncProgressCounter(AsyncProgressHandler):  # callback function
    def _done(self):
        super()._done()
        tqdm.write(os.linesep)

        h, m, s = map(lambda x: int(x), [self.elapsed_time / 3600, self.elapsed_time % 3600 / 60, self.elapsed_time % 60])
        if h == 0 and m == 0:
            etime = f'{s:2d}s'
        elif h == 0:
            etime = f'{m:2d}m {s:2d}s'
        else:
            etime = f'{h}h {m:2d}m {s:2d}s'
        tqdm.write(f"done (elapsed time: {etime})")

    def _update(self):
        super()._update()
        if self.ready_to_update:
            tqdm.write(f'finished {int(self.p)}% of {self.total} iterations ... \r', end="")


class SingletonAsyncProgressCounter(AsyncProgressCounter, metaclass=SingletonType):
    pass


class SingletonNullMsgHandler(metaclass=SingletonType):
    def __call__(self, *args, **kwargs):
        pass


class SingletonMessageHandler(MessageHandler, metaclass=SingletonType):
    pass


class SingletonLogMessage(LogMessage, metaclass=SingletonType):
    pass


class AsyncProgressBar(AsyncProgressHandler):

    def _init(self, total, step):
        super()._init(total, step)
        self.pbar = tqdm(total=total, desc="finished", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    def _done(self):
        self.pbar.close()

    def _update(self):
        super()._update()
        self.pbar.update()


# **************************++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def default_singleton_callback(msg):
    return SingletonMessageHandler([SingletonLogMessage(), SingletonAsyncProgressCounter()])(msg)


def singleton_null_callback(msg):
    return SingletonNullMsgHandler()(msg)
