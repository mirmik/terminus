"""PyQt5-based window backend using :class:`QOpenGLWidget`."""

from __future__ import annotations

from typing import Callable, Optional, Any

from PyQt5 import QtCore, QtGui, QtWidgets

from .base import Action, BackendWindow, Key, MouseButton, WindowBackend


def _qt_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _translate_mouse(button: QtCore.Qt.MouseButton) -> MouseButton:
    if button == QtCore.Qt.LeftButton:
        return MouseButton.LEFT
    if button == QtCore.Qt.RightButton:
        return MouseButton.RIGHT
    if button == QtCore.Qt.MiddleButton:
        return MouseButton.MIDDLE
    return MouseButton.LEFT


def _translate_action(action: bool) -> Action:
    return Action.PRESS if action else Action.RELEASE


def _translate_key(key: int) -> Key:
    if key == QtCore.Qt.Key_Escape:
        return Key.ESCAPE
    if key == QtCore.Qt.Key_Space:
        return Key.SPACE
    try:
        return Key(key)
    except ValueError:
        return Key.UNKNOWN


class _QtGLWidget(QtWidgets.QOpenGLWidget):
    """Thin wrapper forwarding Qt events into backend callbacks."""

    def __init__(self, owner: "QtGLWindowHandle"):
        super().__init__(parent=None)
        self._owner = owner
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        cb = self._owner._framebuffer_callback
        if cb:
            size = event.size()
            cb(self._owner, size.width(), size.height())

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        cb = self._owner._mouse_callback
        if cb:
            cb(self._owner, _translate_mouse(event.button()), Action.PRESS, int(event.modifiers()))

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        cb = self._owner._mouse_callback
        if cb:
            cb(self._owner, _translate_mouse(event.button()), Action.RELEASE, int(event.modifiers()))

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        cb = self._owner._cursor_callback
        if cb:
            cb(self._owner, float(event.x()), float(event.y()))

    def wheelEvent(self, event: QtGui.QWheelEvent):
        cb = self._owner._scroll_callback
        if cb:
            angle = event.angleDelta()
            cb(self._owner, angle.x() / 120.0, angle.y() / 120.0)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        cb = self._owner._key_callback
        if cb:
            cb(self._owner, _translate_key(event.key()), event.nativeScanCode(), Action.PRESS, int(event.modifiers()))

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        cb = self._owner._key_callback
        if cb:
            cb(self._owner, _translate_key(event.key()), event.nativeScanCode(), Action.RELEASE, int(event.modifiers()))


class QtGLWindowHandle(BackendWindow):
    def __init__(self, width: int, height: int, title: str, share: Optional[BackendWindow] = None, parent: Optional[QtWidgets.QWidget] = None):
        self.app = _qt_app()
        self._widget = _QtGLWidget(self)
        self._widget.setWindowTitle(title)
        if parent is not None:
            self._widget.setParent(parent)
        self._widget.resize(width, height)
        self._widget.show()
        self._closed = False
        self._user_ptr: Any = None
        self._framebuffer_callback: Optional[Callable] = None
        self._cursor_callback: Optional[Callable] = None
        self._scroll_callback: Optional[Callable] = None
        self._mouse_callback: Optional[Callable] = None
        self._key_callback: Optional[Callable] = None

    # BackendWindow interface --------------------------------------------
    def close(self):
        if self._closed:
            return
        self._closed = True
        self._widget.close()

    def should_close(self) -> bool:
        return self._closed or not self._widget.isVisible()

    def make_current(self):
        self._widget.makeCurrent()

    def swap_buffers(self):
        ctx = self._widget.context()
        if ctx is not None:
            ctx.swapBuffers(ctx.surface())

    def framebuffer_size(self):
        ratio = float(self._widget.devicePixelRatioF())
        return int(self._widget.width() * ratio), int(self._widget.height() * ratio)

    def window_size(self):
        return self._widget.width(), self._widget.height()

    def get_cursor_pos(self):
        pos = self._widget.mapFromGlobal(QtGui.QCursor.pos())
        return float(pos.x()), float(pos.y())

    def set_should_close(self, flag: bool):
        if flag:
            self.close()

    def set_user_pointer(self, ptr: Any):
        self._user_ptr = ptr

    def set_framebuffer_size_callback(self, callback: Callable):
        self._framebuffer_callback = callback

    def set_cursor_pos_callback(self, callback: Callable):
        self._cursor_callback = callback

    def set_scroll_callback(self, callback: Callable):
        self._scroll_callback = callback

    def set_mouse_button_callback(self, callback: Callable):
        self._mouse_callback = callback

    def set_key_callback(self, callback: Callable):
        self._key_callback = callback

    # Convenience to access underlying widget from examples
    @property
    def widget(self) -> _QtGLWidget:
        return self._widget


class QtWindowBackend(WindowBackend):
    """Window backend relying on PyQt5 for event loop and GL surface."""

    def __init__(self, app: Optional[QtWidgets.QApplication] = None):
        self.app = app or _qt_app()

    def create_window(self, width: int, height: int, title: str, share: Optional[BackendWindow] = None, parent: Optional[QtWidgets.QWidget] = None) -> QtGLWindowHandle:
        return QtGLWindowHandle(width, height, title, share=share, parent=parent)

    def poll_events(self):
        # process pending events without blocking
        self.app.processEvents()

    def terminate(self):
        self.app.quit()
