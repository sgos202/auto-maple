"""
A transparent, click-through window drawn on top of the game for showing what a
solver sees (boxes and labels). It never takes focus, passes every click to the
game underneath, and is excluded from screen capture so the solvers' own
screenshots don't contain it.

Create it and call ``show`` from one thread only.
"""

import ctypes
from ctypes import wintypes


GREEN = 0x0000FF00      # COLORREF is 0x00BBGGRR
RED = 0x000000FF
YELLOW = 0x0000FFFF
CYAN = 0x00FFFF00
MAGENTA = 0x00FF00FF
WHITE = 0x00FFFFFF
PINK = 0x00B469FF       # hot pink, RGB(255, 105, 180)

_WS_POPUP = 0x80000000
_WS_EX = 0x00080000 | 0x00000020 | 0x00000008 | 0x00000080 | 0x08000000
#        layered      transparent  topmost      toolwindow   noactivate
_LWA_COLORKEY = 0x1
_WDA_EXCLUDEFROMCAPTURE = 0x11
_SWP_NOACTIVATE = 0x0010
_SW_SHOWNOACTIVATE = 4
_SRCCOPY = 0x00CC0020
_KEY = 0x00000000       # black is the see-through colour

_WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                              wintypes.WPARAM, wintypes.LPARAM)


class _WNDCLASS(ctypes.Structure):
    _fields_ = [('style', wintypes.UINT), ('lpfnWndProc', _WNDPROC),
                ('cbClsExtra', ctypes.c_int), ('cbWndExtra', ctypes.c_int),
                ('hInstance', wintypes.HINSTANCE), ('hIcon', wintypes.HANDLE),
                ('hCursor', wintypes.HANDLE), ('hbrBackground', wintypes.HANDLE),
                ('lpszMenuName', wintypes.LPCWSTR), ('lpszClassName', wintypes.LPCWSTR)]


def _api():
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    H, I, U = wintypes.HANDLE, ctypes.c_int, wintypes.UINT
    signatures = [
        (user32, 'DefWindowProcW', ctypes.c_ssize_t, [wintypes.HWND, U, wintypes.WPARAM, wintypes.LPARAM]),
        (user32, 'RegisterClassW', wintypes.ATOM, [ctypes.POINTER(_WNDCLASS)]),
        (user32, 'CreateWindowExW', wintypes.HWND,
         [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, I, I, I, I, H, H, H, H]),
        (user32, 'DestroyWindow', wintypes.BOOL, [wintypes.HWND]),
        (user32, 'SetLayeredWindowAttributes', wintypes.BOOL, [wintypes.HWND, wintypes.DWORD, ctypes.c_byte, wintypes.DWORD]),
        (user32, 'SetWindowDisplayAffinity', wintypes.BOOL, [wintypes.HWND, wintypes.DWORD]),
        (user32, 'SetWindowPos', wintypes.BOOL, [wintypes.HWND, wintypes.HWND, I, I, I, I, U]),
        (user32, 'ShowWindow', wintypes.BOOL, [wintypes.HWND, I]),
        (user32, 'PeekMessageW', wintypes.BOOL, [ctypes.POINTER(wintypes.MSG), wintypes.HWND, U, U, U]),
        (user32, 'TranslateMessage', wintypes.BOOL, [ctypes.POINTER(wintypes.MSG)]),
        (user32, 'DispatchMessageW', ctypes.c_ssize_t, [ctypes.POINTER(wintypes.MSG)]),
        (user32, 'GetDC', H, [wintypes.HWND]),
        (user32, 'ReleaseDC', I, [wintypes.HWND, H]),
        (user32, 'FillRect', I, [H, ctypes.POINTER(wintypes.RECT), H]),
        (gdi32, 'CreateCompatibleDC', H, [H]),
        (gdi32, 'CreateCompatibleBitmap', H, [H, I, I]),
        (gdi32, 'SelectObject', H, [H, H]),
        (gdi32, 'DeleteObject', wintypes.BOOL, [H]),
        (gdi32, 'DeleteDC', wintypes.BOOL, [H]),
        (gdi32, 'CreatePen', H, [I, I, wintypes.DWORD]),
        (gdi32, 'CreateSolidBrush', H, [wintypes.DWORD]),
        (gdi32, 'GetStockObject', H, [I]),
        (gdi32, 'Rectangle', wintypes.BOOL, [H, I, I, I, I]),
        (gdi32, 'Polyline', wintypes.BOOL, [H, ctypes.POINTER(wintypes.POINT), I]),
        (gdi32, 'SetTextColor', wintypes.DWORD, [H, wintypes.DWORD]),
        (gdi32, 'SetBkColor', wintypes.DWORD, [H, wintypes.DWORD]),
        (gdi32, 'TextOutW', wintypes.BOOL, [H, I, I, wintypes.LPCWSTR, I]),
        (gdi32, 'BitBlt', wintypes.BOOL, [H, I, I, I, I, H, I, I, wintypes.DWORD]),
        (gdi32, 'CreateFontW', H, [I, I, I, I, I, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.DWORD, wintypes.LPCWSTR]),
        (kernel32, 'GetModuleHandleW', H, [wintypes.LPCWSTR]),
    ]
    for library, name, restype, argtypes in signatures:
        function = getattr(library, name)
        function.restype, function.argtypes = restype, argtypes
    return user32, gdi32, kernel32


class ScreenOverlay:
    _class_registered = False
    _wndproc = None     # kept alive for as long as the window class exists

    def __init__(self):
        self.user32, self.gdi32, kernel32 = _api()
        instance = kernel32.GetModuleHandleW(None)
        name = 'AutoMapleSolverOverlay'
        if not ScreenOverlay._class_registered:
            ScreenOverlay._wndproc = _WNDPROC(self.user32.DefWindowProcW)
            window_class = _WNDCLASS()
            window_class.lpfnWndProc = ScreenOverlay._wndproc
            window_class.hInstance = instance
            window_class.lpszClassName = name
            if not self.user32.RegisterClassW(ctypes.byref(window_class)):
                raise ctypes.WinError(ctypes.get_last_error())
            ScreenOverlay._class_registered = True
        self.handle = self.user32.CreateWindowExW(_WS_EX, name, 'overlay', _WS_POPUP,
                                                  0, 0, 1, 1, None, None, instance, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.user32.SetLayeredWindowAttributes(self.handle, _KEY, 0, _LWA_COLORKEY)
        if not self.user32.SetWindowDisplayAffinity(self.handle, _WDA_EXCLUDEFROMCAPTURE):
            # Without this the boxes would appear in the solver's own screenshots.
            self.close()
            raise OSError('this Windows version cannot hide the overlay from screen capture')
        # Pixel font without smoothing: smoothed edges would blend with the colour key.
        self.font = self.gdi32.CreateFontW(-16, 0, 0, 0, 700, 0, 0, 0, 1, 0, 0, 3, 0, 'Consolas')
        self.visible = False

    def show(self, region, boxes=(), labels=(), outlines=(), circles=()):
        """
        Covers REGION ({'left', 'top', 'width', 'height'} in screen pixels) and draws
        BOXES [(x0, y0, x1, y1, colour, thickness)], LABELS [(x, y, text, colour)]
        OUTLINES [(points [(x, y), ...], colour, thickness)] (closed shapes) and
        CIRCLES [(x, y, radius, colour, thickness)],
        all relative to the region's top-left.
        """
        if not self.handle:
            return
        user32, gdi32 = self.user32, self.gdi32
        width, height = max(1, region['width']), max(1, region['height'])
        user32.SetWindowPos(self.handle, wintypes.HWND(-1), region['left'], region['top'],
                            width, height, _SWP_NOACTIVATE)
        if not self.visible:
            user32.ShowWindow(self.handle, _SW_SHOWNOACTIVATE)
            self.visible = True

        screen = user32.GetDC(self.handle)
        memory = gdi32.CreateCompatibleDC(screen)
        bitmap = gdi32.CreateCompatibleBitmap(screen, width, height)
        old_bitmap = gdi32.SelectObject(memory, bitmap)
        background = gdi32.CreateSolidBrush(_KEY)
        user32.FillRect(memory, ctypes.byref(wintypes.RECT(0, 0, width, height)), background)
        gdi32.DeleteObject(background)

        gdi32.SelectObject(memory, gdi32.GetStockObject(5))     # hollow brush
        for x0, y0, x1, y1, colour, thickness in boxes:
            pen = gdi32.CreatePen(0, max(1, int(thickness)), colour)
            old_pen = gdi32.SelectObject(memory, pen)
            gdi32.Rectangle(memory, int(x0), int(y0), int(x1), int(y1))
            gdi32.SelectObject(memory, old_pen)
            gdi32.DeleteObject(pen)
        for points, colour, thickness in outlines:
            if len(points) < 2:
                continue
            closed = list(points) + [points[0]]
            array = (wintypes.POINT * len(closed))(
                *[wintypes.POINT(int(x), int(y)) for x, y in closed])
            pen = gdi32.CreatePen(0, max(1, int(thickness)), colour)
            old_pen = gdi32.SelectObject(memory, pen)
            gdi32.Polyline(memory, array, len(closed))
            gdi32.SelectObject(memory, old_pen)
            gdi32.DeleteObject(pen)
        for x, y, radius, colour, thickness in circles:
            pen = gdi32.CreatePen(0, max(1, int(thickness)), colour)
            old_pen = gdi32.SelectObject(memory, pen)
            gdi32.Ellipse(memory, int(x - radius), int(y - radius),
                          int(x + radius), int(y + radius))
            gdi32.SelectObject(memory, old_pen)
            gdi32.DeleteObject(pen)
        old_font = gdi32.SelectObject(memory, self.font)
        gdi32.SetBkColor(memory, 0x00010101)    # near-black, so labels stay readable
        for x, y, text, colour in labels:
            gdi32.SetTextColor(memory, colour)
            gdi32.TextOutW(memory, int(x), int(y), text, len(text))
        gdi32.SelectObject(memory, old_font)

        gdi32.BitBlt(screen, 0, 0, width, height, memory, 0, 0, _SRCCOPY)
        gdi32.SelectObject(memory, old_bitmap)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory)
        user32.ReleaseDC(self.handle, screen)
        self._pump()

    def _pump(self):
        message = wintypes.MSG()
        while self.user32.PeekMessageW(ctypes.byref(message), self.handle, 0, 0, 1):
            self.user32.TranslateMessage(ctypes.byref(message))
            self.user32.DispatchMessageW(ctypes.byref(message))

    def close(self):
        if self.handle:
            self.user32.DestroyWindow(self.handle)
            self.handle = None
        if getattr(self, 'font', None):
            self.gdi32.DeleteObject(self.font)
            self.font = None
