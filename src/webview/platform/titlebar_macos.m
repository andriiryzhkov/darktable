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

#import <Cocoa/Cocoa.h>
#include <webview/api.h>

static NSWindow *_get_window(webview_t w)
{
  return (__bridge NSWindow *)webview_get_native_handle(
      w, WEBVIEW_NATIVE_HANDLE_KIND_UI_WINDOW);
}

/* Reposition the native traffic light buttons (close/minimize/zoom)
 * to be vertically centered within our 52px headerbar, with reasonable
 * left margin matching macOS conventions.
 *
 * Uses Auto Layout constraints so positions survive resize/fullscreen. */
static void _constrain_traffic_lights(NSWindow *window)
{
  static const CGFloat kHeaderHeight = 52.0;
  static const CGFloat kLeftMargin   = 13.0;  // standard macOS left inset
  static const CGFloat kButtonGap    =  6.0;  // horizontal spacing between buttons

  NSButton *buttons[3] = {
    [window standardWindowButton:NSWindowCloseButton],
    [window standardWindowButton:NSWindowMiniaturizeButton],
    [window standardWindowButton:NSWindowZoomButton],
  };
  if(!buttons[0] || !buttons[1] || !buttons[2]) return;

  NSView *container = buttons[0].superview;
  if(!container) return;

  CGFloat btnH = buttons[0].frame.size.height;
  CGFloat topInset = (kHeaderHeight - btnH) / 2.0;

  CGFloat x = kLeftMargin;
  for(int i = 0; i < 3; i++)
  {
    NSButton *btn = buttons[i];
    btn.translatesAutoresizingMaskIntoConstraints = NO;

    /* Pin to top of superview with inset to vertically center in headerbar */
    [NSLayoutConstraint activateConstraints:@[
      [btn.topAnchor constraintEqualToAnchor:container.topAnchor constant:topInset],
      [btn.leadingAnchor constraintEqualToAnchor:container.leadingAnchor constant:x],
      [btn.widthAnchor constraintEqualToConstant:btn.frame.size.width],
      [btn.heightAnchor constraintEqualToConstant:btnH],
    ]];
    x += btn.frame.size.width + kButtonGap;
  }
}

void dt_titlebar_init(webview_t w)
{
  NSWindow *window = _get_window(w);

  // extend content view to fill entire window including titlebar area
  window.styleMask |= NSWindowStyleMaskFullSizeContentView;

  // make titlebar background transparent — traffic lights remain visible
  window.titlebarAppearsTransparent = YES;

  // hide the title text (we show our own header)
  window.titleVisibility = NSWindowTitleHidden;

  // center traffic lights vertically in our headerbar via Auto Layout
  _constrain_traffic_lights(window);
}

void dt_titlebar_start_drag(webview_t w)
{
  NSWindow *window = _get_window(w);
  NSEvent *event = [NSApp currentEvent];
  if(event)
    [window performWindowDragWithEvent:event];
}

void dt_titlebar_zoom(webview_t w)
{
  // respects system preferences (zoom vs minimize on double-click)
  NSWindow *window = _get_window(w);
  [window zoom:nil];
}
