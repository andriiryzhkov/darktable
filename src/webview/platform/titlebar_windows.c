/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <windows.h>
#include <dwmapi.h>
#include <webview/api.h>

static HWND g_hwnd = NULL;
static WNDPROC g_orig_proc = NULL;

static LRESULT CALLBACK _titlebar_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
  if(msg == WM_NCCALCSIZE && wp == TRUE)
  {
    // remove titlebar height from non-client area
    // native caption buttons are still drawn by DWM in top-right
    NCCALCSIZE_PARAMS *params = (NCCALCSIZE_PARAMS *)lp;
    params->rgrc[0].top += 1; // 1px for DWM shadow to work
    return 0;
  }

  return CallWindowProc(g_orig_proc, hwnd, msg, wp, lp);
}

void dt_titlebar_init(webview_t w)
{
  g_hwnd = (HWND)webview_get_native_handle(
      w, WEBVIEW_NATIVE_HANDLE_KIND_UI_WINDOW);

  // extend DWM frame — preserves window shadow and native controls
  MARGINS margins = { 0, 0, 1, 0 };
  DwmExtendFrameIntoClientArea(g_hwnd, &margins);

  // subclass window procedure
  g_orig_proc = (WNDPROC)SetWindowLongPtr(
      g_hwnd, GWLP_WNDPROC, (LONG_PTR)_titlebar_wndproc);

  // force frame recalculation
  SetWindowPos(g_hwnd, NULL, 0, 0, 0, 0,
      SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER);
}

void dt_titlebar_start_drag(webview_t w)
{
  (void)w;
  // tell Windows to start a titlebar drag
  ReleaseCapture();
  SendMessage(g_hwnd, WM_NCLBUTTONDOWN, HTCAPTION, 0);
}

void dt_titlebar_zoom(webview_t w)
{
  (void)w;
  ShowWindow(g_hwnd, IsZoomed(g_hwnd) ? SW_RESTORE : SW_MAXIMIZE);
}
