#!/usr/bin/env bash
#
# Build and install an official darktable release on Linux.
#
#   ./install_release.sh                  # latest release into /opt/darktable
#   ./install_release.sh release-5.6.0    # a specific tag
#   ./install_release.sh --list           # what releases are available
#
# Or run it straight from GitHub without downloading it first. The "-s --" is
# what carries the options past bash to the script:
#
#   curl -fsSL https://raw.githubusercontent.com/andriiryzhkov/darktable/refs/heads/install_tools/tools/install_release.sh | bash -s -- --with-ai
#
# It still asks before installing packages or replacing an install, because it
# prompts on /dev/tty rather than stdin, which is the script itself when piped.
# Add --yes for an unattended run.
#
# Options may follow the tag, or precede it. Any option below is this script's;
# every other flag is passed on to build.sh, so --disable-opencl and friends
# work, and everything after -- goes on to cmake. build.sh's --install and
# --sudo are refused: this script decides when to install and when to use root.
#
#   --user           install under $HOME, needing no root at all
#   --keep-source    keep the source tree, so building another release later
#                    only recompiles what changed
#   --with-ai        build with AI support
#   --skip-deps      do not touch the package manager
#   --skip-lensfun   do not update the lensfun lens database
#   --desktop-only   only refresh the menu entry and icons, build nothing
#   --clean          remove source trees left behind by failed builds
#   --uninstall      remove an install made by this script
#   --list           print the recent releases and stop
#   --help           print this comment and stop
#   --prefix DIR     install somewhere other than /opt/darktable. DIR has to be
#                    an absolute path, a directory of its own, and one that is
#                    empty or already holds a darktable: a top-level directory,
#                    anything under /usr except /usr/local, your home and your
#                    ~/.local are refused, when installing as well as when
#                    uninstalling
#   --yes            do not ask before installing packages or replacing an
#                    install
#
# DT_SRC names a source tree to keep and reuse, DT_LINKDIR where the symlinks
# go (default /usr/local/bin, or ~/.local/bin under --user).
#
# git, curl, cmake and realpath have to be there already; everything else the
# build needs is installed for you unless --skip-deps.
#
# Most people do not need this: an AppImage, a flatpak or your distribution's
# package gives you a working darktable with no toolchain. Build from source
# when you want the pixel pipeline compiled for your own CPU - the published
# binaries are built -mtune=generic so they run anywhere, while this gets
# -march=native (cmake/march-mtune.cmake) - or native Wayland, your system GTK
# theme, or options the official build does not carry.
#
# WHERE IT INSTALLS
#
# By default everything is system-wide and needs root: the prefix is
# /opt/darktable, binaries are linked into /usr/local/bin, and the desktop entry
# and icons into /usr/share. --user puts all of it under your home directory
# instead - ~/.local/darktable, ~/.local/bin, ~/.local/share - and needs no root
# at all, though the dependencies still do. The prefix is a directory of its own
# in both cases rather than merging into /usr/local or ~/.local, so it can be
# removed in one go.
#
# THE SOURCE TREE
#
# By default the source is fetched into a scratch directory under ~/.cache and
# removed once the install succeeds. A clone and build want around 800 MB, and
# ~/.cache rather than /tmp because /tmp is often tmpfs. A failed build keeps
# its tree and prints the path, so you can see what went wrong; --clean removes
# those afterwards. --keep-source keeps it at ~/src/darktable-release instead
# and reuses it. The clone is blobless (--filter=blob:none), around 50 MB rather
# than the gigabytes a full history costs, while still carrying every tag.
#
# DEPENDENCIES
#
# The Debian and Ubuntu list is the one darktable's own CI installs, from
# .github/workflows/ci.yml, plus liblensfun-bin for the lens database update.
# The Fedora and Arch lists are a best-effort mapping and are NOT covered by CI:
# a name a distribution does not carry is skipped rather than failing the run,
# but a genuinely missing dependency then surfaces later as a cmake error.
# Corrections welcome. Any other distribution: install them yourself and use
# --skip-deps.
#
# REMOVING IT
#
# --uninstall offers whichever installs it finds in /opt/darktable and
# ~/.local/darktable, unless --user or --prefix names one. It also works when
# the prefix is already gone, clearing the symlinks a failed install left
# pointing at nothing - that case has no prefix to find, so it needs --prefix,
# and a failed run prints the exact command. Note that --prefix
# on its own only moves the prefix: the symlinks and the desktop entry still go
# to the system-wide locations unless --user comes with it.
#
# A prefix that is neither empty nor a darktable is refused rather than removed,
# as is any of the directories --prefix above will not accept. That stops the
# accidents worth stopping, not a determined one: an empty directory meeting
# those rules is still removed, so --prefix is not a safe thing to point at
# something you want kept. Only symlinks that resolve into the prefix are
# removed, and a file or symlink moved aside as .bak at install time is put
# back, unless something has since put a real file back in its place - a
# distribution package upgrade does that, and the newer file wins. Your
# settings and library in ~/.config/darktable are never touched.

set -euo pipefail

# every assumption below - the package managers, /usr/share, the desktop and
# icon caches - is a Linux one. macOS has its own packaging and Windows builds
# through MSYS2, so fail here rather than half way through
case "$(uname -s)" in
  Linux) ;;
  Darwin) printf 'this script is Linux only; on macOS use the .dmg or homebrew\n' >&2; exit 1 ;;
  *)      printf 'this script is Linux only (found %s)\n' "$(uname -s)" >&2; exit 1 ;;
esac

REPO="https://github.com/darktable-org/darktable.git"
API="https://api.github.com/repos/darktable-org/darktable/releases"
# DT_SRC names a tree to keep and reuse. with neither it nor --keep-source the
# source is fetched into a scratch directory and removed once installed.
# under $HOME rather than /tmp: a clone and build want ~800M, and /tmp is
# frequently tmpfs, so this would otherwise be ~800M of RAM
SRC="${DT_SRC:-}"
CACHE=""    # set in main, once HOME is canonical
SCRATCH=""
PREFIX="" LINKDIR="" DATADIR=""
USER_MODE=0
WITH_AI=0 SKIP_DEPS=0 SKIP_LENSFUN=0 ASSUME_YES=0 KEEP_SRC=0
ACTION=install
EXPLICIT_TARGET=0
TAG="" PASSTHROUGH=()

# Debian and Ubuntu packages, taken from what darktable's own CI installs
# (.github/workflows/ci.yml). Other distributions are a best-effort mapping
# and are not covered by CI - see the DEPENDENCIES section above
DEB_PACKAGES="
  appstream-util build-essential cmake desktop-file-utils
  gdb gettext git intltool
  libarchive-dev libatk1.0-dev libavif-dev libcairo2-dev
  libcmocka-dev libcolord-dev libcolord-gtk-dev libcups2-dev
  libcurl4-gnutls-dev libexiv2-dev libgdk-pixbuf-2.0-dev libglib2.0-dev
  libgmic-dev libgphoto2-dev libgraphicsmagick1-dev libgtk-3-dev
  libheif-dev libjpeg-dev libjson-glib-dev liblcms2-dev
  liblensfun-dev liblensfun-bin liblua5.4-dev libonnxruntime-dev
  libopencv-calib3d-dev
  libopencv-core-dev libopencv-features2d-dev libopencv-flann-dev libopencv-imgproc-dev
  libopenexr-dev libopenjp2-7-dev libosmgpsmap-1.0-dev libpango1.0-dev
  libpng-dev libportmidi-dev libpotrace-dev libpugixml-dev
  libraw-dev librsvg2-dev libsaxon-java libsdl2-dev
  libsecret-1-dev libsqlite3-dev libtiff5-dev libwebp-dev
  libx11-dev libxml2-dev libxml2-utils ninja-build
  perl po4a python3-jsonschema xsltproc zlib1g-dev
"

RPM_PACKAGES="
  gcc gcc-c++ cmake ninja-build git gettext intltool
  desktop-file-utils libappstream-glib perl po4a
  cairo-devel colord-devel colord-gtk-devel cups-devel exiv2-devel
  gtk3-devel gmic-devel GraphicsMagick-devel LibRaw-devel lcms2-devel
  lensfun lensfun-devel libavif-devel libcurl-devel libgphoto2-devel libheif-devel
  libjpeg-turbo-devel libomp-devel libpng-devel librsvg2-devel libsecret-devel
  libtiff-devel libwebp-devel libxml2-devel libxslt json-glib-devel
  lua-devel opencv-devel OpenEXR-devel openjpeg2-devel osm-gps-map-devel
  portmidi-devel potrace-devel pugixml-devel SDL2-devel sqlite-devel
  zlib-devel libcmocka-devel python3-jsonschema
"

ARCH_PACKAGES="
  base-devel cmake ninja git gettext intltool
  desktop-file-utils appstream-glib perl po4a
  cairo colord colord-gtk cups exiv2 gtk3 gmic graphicsmagick libraw lcms2
  lensfun libavif curl libgphoto2 libheif libjpeg-turbo libpng librsvg
  libsecret libtiff libwebp libxml2 libxslt json-glib lua opencv openexr
  openjpeg2 osm-gps-map portmidi potrace pugixml sdl2 sqlite zlib cmocka
  python-jsonschema
"

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

cleanup() {
  local rc=$?
  [ -n "$SCRATCH" ] && [ -d "$SCRATCH" ] || return 0
  if [ "$rc" -eq 0 ]; then
    rm -rf -- "$SCRATCH"
  else
    printf 'the source tree is at %s if you want to look at it\n' "$SCRATCH" >&2
  fi
}
trap cleanup EXIT
note() { printf '\033[1m==>\033[0m %s\n' "$*"; }

ask() {
  [ "$ASSUME_YES" = 1 ] && return 0
  # not [ -t 0 ]: piped from curl stdin is the script itself, yet /dev/tty is
  # still there to ask on. and not [ -r /dev/tty ] either: the device node is
  # readable under cron or docker without -t, where opening it still fails, so
  # try the open itself
  { : </dev/tty; } 2>/dev/null ||
    die "nothing to prompt on; re-run with --yes to accept: $1"
  local reply=""
  printf '%s [y/N] ' "$1"
  read -r reply </dev/tty || true
  case "$reply" in [yY]*) return 0 ;; *) return 1 ;; esac
}

sudo_() {
  if [ "$USER_MODE" = 1 ] || [ "$(id -u)" = 0 ]; then "$@"; else sudo "$@"; fi
}

# a build that failed left its tree behind on purpose, so it could be looked
# at. this is how you get rid of them afterwards
clean_scratch() {
  local found=0 d
  for d in "$CACHE"/darktable-install.*; do
    [ -d "$d" ] || continue          # no match leaves the glob unexpanded
    printf '  %s  %s\n' "$(du -sh "$d" 2>/dev/null | cut -f1)" "$d"
    found=$((found + 1))
  done

  if [ "$found" = 0 ]; then
    note "no leftover source trees in $CACHE"
  else
    ask "remove these $found source trees?" || die "nothing done"
    rm -rf -- "$CACHE"/darktable-install.*
    note "removed $found"
  fi

  # the kept tree is deliberate, so say it is there rather than delete it
  [ -d "$HOME/src/darktable-release" ] &&
    note "note: the --keep-source tree at ~/src/darktable-release ($(du -sh "$HOME/src/darktable-release" 2>/dev/null | cut -f1)) is left alone"
  return 0
}

# the canonical path: a trailing slash, a doubled slash, a .. and every
# symlinked component resolved. -m so that a path whose parent no longer exists
# still resolves, which is the state a failed install leaves its links in
_canon() {
  realpath -m -- "$1"
}

# true when $1, canonical and compared against a canonical HOME, is a directory
# this script may create and destroy
_is_private_prefix() {
  # one component deep is a top-level directory: nothing this script owns lives
  # there, and a typo in one should not be removable
  case "${1#/}" in */*) ;; *) return 1 ;; esac
  case "$1" in
    # /usr is the distribution's, but /usr/local is the local admin's and
    # /usr/local/darktable is the most ordinary choice after /opt
    /usr/local)   return 1 ;;
    /usr/local/*) ;;
    /usr|/usr/*)  return 1 ;;
  esac
  # a home directory itself, and the ~/.local everything on the machine shares
  case "$1" in "$HOME"|"$HOME/.local") return 1 ;; esac
  return 0
}

# -L as well as -e: a failed install can leave bin/darktable dangling, and the
# tool that made the mess should still be able to clear it up
_looks_like_darktable() {
  [ -e "$1/bin/darktable" ] || [ -L "$1/bin/darktable" ] ||
    [ -d "$1/share/darktable" ]
}

# why this prefix may not be created or destroyed, on stdout, with a non-zero
# status. callers given the path explicitly turn that into a die; the discovery
# loop skips instead, so one unrecognized directory cannot strand the other
# install half removed
_prefix_objection() {
  _is_private_prefix "$1" ||
    { printf 'that is not a private install prefix'; return 1; }
  [ -e "$1" ] || return 0
  _looks_like_darktable "$1" && return 0
  # empty is safe to remove, and people pre-create the prefix so that the
  # install needs no root. a dedicated mount is never literally empty, hence
  # lost+found. the -d and -r -x tests come first because find prints nothing
  # for a regular file and nothing for a directory it may not read, and both
  # would otherwise pass for empty and be handed to rm -rf
  [ -d "$1" ] ||
    { printf 'that is not a directory'; return 1; }
  [ -r "$1" ] && [ -x "$1" ] ||
    { printf 'it cannot be read, so it cannot be shown to be empty'; return 1; }
  [ -z "$(find "$1" -mindepth 1 -maxdepth 1 ! -name lost+found -print -quit)" ] ||
    { printf 'it is neither empty nor a darktable install'; return 1; }
}

# a backup goes back only over a destination this uninstall vacated: one that
# is gone, or still holds a symlink of ours into the prefix. an upgrade may
# have put a fresh real file there in the meantime, and that one is newer than
# the backup, so overwriting it would destroy exactly what the backup exists
# to protect
_restore_bak() {
  local bak="$1" prefix="$2" dst="${1%.bak}"
  [ -e "$bak" ] || [ -L "$bak" ] || return 0
  if [ -L "$dst" ]; then
    case "$(_canon "$dst")" in "$prefix"/*) sudo_ rm -f -- "$dst" ;; *) return 0 ;; esac
  elif [ -e "$dst" ]; then
    note "not restoring $dst: something else has put a real file back there"
    return 0
  fi
  sudo_ mv -f -- "$bak" "$dst"
  note "restored $dst"
}

# without these the menu entry can take a login to appear, and show no icon
_refresh_caches() {
  local datadir="$1"
  command -v update-desktop-database >/dev/null &&
    sudo_ update-desktop-database -q "$datadir/applications" || true
  # -t because only the distro ships an index.theme, in /usr/share/icons/hicolor:
  # a --user install writes into ~/.local/share/icons/hicolor, which has none, and
  # without it the cache is never built and the error escapes -q
  command -v gtk-update-icon-cache >/dev/null && [ -d "$datadir/icons/hicolor" ] &&
    sudo_ gtk-update-icon-cache -q -t -f "$datadir/icons/hicolor" || true
}

# remove one install and nothing else. only symlinks that actually point into
# the prefix are followed, so a link to another darktable is left alone
_uninstall_one() {
  local raw="$1" linkdir="$2" datadir="$3" as_user="$4"
  local saved="$USER_MODE"
  USER_MODE="$as_user"          # a user install needs no root to remove

  # every path is compared canonical against canonical, or a prefix spelled
  # with a trailing slash or reached through a symlink matches none of its own
  # links and they are all left dangling. uninstall() has vetted the prefix
  local prefix; prefix="$(_canon "$raw")"
  linkdir="$(_canon "$linkdir")"; datadir="$(_canon "$datadir")"
  # test -L follows a trailing slash, so strip them before asking
  while [ "$raw" != "/" ] && [ "$raw" != "${raw%/}" ]; do raw="${raw%/}"; done

  # empty arrays expand to nothing under set -u on bash 4.4 and later, which is
  # everything this script can run on, so none of them need guarding
  local links=() link bak
  for link in "$linkdir"/* "$datadir"/applications/*darktable* \
              "$datadir"/icons/hicolor/*/apps/darktable*; do
    [ -L "$link" ] || continue
    case "$(_canon "$link")" in "$prefix"/*) links+=("$link") ;; esac
  done

  for link in "${links[@]}"; do sudo_ rm -f -- "$link"; done
  # only when this run took something away: a prefix that was never here has no
  # business restoring backups it did not make, which a nonexistent --prefix
  # would otherwise do to any darktable .bak sharing the linkdir.
  # the .bak files are globbed rather than the paths they shadow: those paths
  # hold either nothing, now that the loop above has deleted our symlinks, or
  # whatever has since taken their place, which _restore_bak decides about
  if [ "${#links[@]}" -gt 0 ]; then
    for bak in "$linkdir"/*darktable*.bak \
               "$datadir/applications/org.darktable.darktable.desktop.bak" \
               "$datadir"/icons/hicolor/*/apps/darktable*.bak; do
      _restore_bak "$bak" "$prefix"
    done
  fi
  local removed=0
  if [ -d "$prefix" ]; then
    sudo_ rm -rf -- "$prefix"
    removed=1
  fi
  # the prefix was reached through a symlink: once the tree is gone, so is the
  # only thing that link was for. outside the branch above, because the pass
  # that removes the tree need not be the pass holding the symlinked name
  [ ! -L "$raw" ] || [ -d "$prefix" ] || sudo_ rm -f -- "$raw"

  _refresh_caches "$datadir"

  if [ "$removed" = 1 ]; then
    note "removed $prefix and ${#links[@]} symlinks"
  else
    note "$prefix was already gone; cleared ${#links[@]} symlinks it left behind"
  fi
  USER_MODE="$saved"
}

# with neither --user nor --prefix, both locations are offered: someone who
# tried one and then the other should not have to remember which
uninstall() {
  # four parallel arrays rather than one of packed records: no delimiter is
  # safe, since every character but / and NUL is legal in a path, and a prefix
  # containing the delimiter would silently unpack as a different directory
  local -a cpre=() cbin=() cdat=() cusr=() fpre=() fbin=() fdat=() fusr=()
  local explicit=0
  if [ "$EXPLICIT_TARGET" = 1 ]; then
    # a prefix that is already gone is still worth running over: a failed
    # install leaves the links behind, and this is what clears them
    explicit=1
    [ -d "$PREFIX" ] ||
      note "nothing at $PREFIX; removing whatever it left behind"
    cpre+=("$PREFIX"); cbin+=("$LINKDIR"); cdat+=("$DATADIR"); cusr+=("$USER_MODE")
  else
    # honor DT_LINKDIR here too, or an install made with it leaves its
    # symlinks behind
    if [ -d "/opt/darktable" ]; then
      cpre+=("/opt/darktable"); cbin+=("${DT_LINKDIR:-/usr/local/bin}")
      cdat+=("/usr/share");     cusr+=(0)
    fi
    if [ -d "$HOME/.local/darktable" ]; then
      cpre+=("$HOME/.local/darktable"); cbin+=("${DT_LINKDIR:-$HOME/.local/bin}")
      cdat+=("${XDG_DATA_HOME:-$HOME/.local/share}"); cusr+=(1)
    fi
  fi

  # vetted before the prompt, so nothing is offered that would then be refused
  local i prefix why
  for ((i = 0; i < ${#cpre[@]}; i++)); do
    prefix="$(_canon "${cpre[i]}")"
    # two names for one install - ~/.local/darktable pointing at /opt/darktable
    # is an ordinary way to give a system install a user-visible name - are
    # deliberately both kept: they carry different linkdirs and datadirs, and
    # collapsing them would sweep only one of the two
    if ! why="$(_prefix_objection "$prefix")"; then
      [ "$explicit" = 0 ] || die "refusing to remove $prefix: $why"
      note "skipping $prefix: $why"
      continue
    fi
    # the name as given is kept: a prefix that is itself a symlink needs the
    # link removed as well as the tree it points at
    fpre+=("${cpre[i]}"); fbin+=("${cbin[i]}"); fdat+=("${cdat[i]}"); fusr+=("${cusr[i]}")
  done
  [ "${#fpre[@]}" -gt 0 ] || die "no darktable install found in /opt or ~/.local"

  note "about to remove:"
  for ((i = 0; i < ${#fpre[@]}; i++)); do
    # canonical, or du reports 0 for a prefix that is a symlink
    prefix="$(_canon "${fpre[i]}")"
    printf '  %s  %s%s\n' "$(du -sh "$prefix" 2>/dev/null | cut -f1)" "$prefix" \
      "$([ "${fusr[i]}" = 1 ] && echo '  (user)' || echo '  (system, needs root)')"
  done
  printf '  your settings and library in ~/.config/darktable are NOT touched\n'
  ask "remove ${#fpre[@]} install(s)?" || die "nothing done"

  for ((i = 0; i < ${#fpre[@]}; i++)); do
    _uninstall_one "${fpre[i]}" "${fbin[i]}" "${fdat[i]}" "${fusr[i]}"
  done
  note "settings and library kept in ~/.config/darktable"
}

# --- releases -------------------------------------------------------------

# the newest release is not the highest version number: darktable's odd minor
# versions are development releases, so 5.7.0 predates 5.6.1. ask GitHub which
# one is flagged latest rather than sorting tags
# the pipelines are captured rather than run bare: under pipefail a grep that
# matches nothing returns 1, which set -e would turn into a silent exit
latest_tag() {
  local json tag
  json="$(curl -fsSL "$API/latest")" || die "could not reach the GitHub API"
  tag="$(printf '%s\n' "$json" | grep '"tag_name"' | head -1 \
    | sed -E 's/.*"tag_name"[^"]*"([^"]+)".*/\1/')" || true
  [ -n "$tag" ] || die "no tag_name in the GitHub API response"
  printf '%s\n' "$tag"
}

list_releases() {
  local json out
  json="$(curl -fsSL "$API?per_page=15")" ||
    die "could not reach the GitHub API"
  out="$(printf '%s\n' "$json" \
    | grep -E '"(tag_name|prerelease)"' | paste - - \
    | sed -nE 's/.*"tag_name"[^"]*"([^"]+)".*"prerelease": (true|false).*/\1 \2/p' \
    | awk '{ printf "%-18s %s\n", $1, ($2=="true" ? "(prerelease)" : "") }')" || true
  [ -n "$out" ] || die "could not read the release list from the GitHub API"
  printf '%s\n' "$out"
}

# --- dependencies ---------------------------------------------------------

# print the names this manager actually carries, one per line. apt-get has no
# tolerant mode at all and pacman -S drops the whole transaction on one unknown
# target, so the list has to be filtered before either of them sees it
_available_packages() {
  local mgr="$1"; shift
  case "$mgr" in
    apt)
      # policy lists only what it knows, and a name with no candidate is
      # virtual: apt-get install would still refuse it.
      # LC_ALL=C because the field names are translated - the German catalog
      # has "Installationskandidat:" and "(keine)" - and matching the English
      # would then drop every package
      LC_ALL=C apt-cache policy -- "$@" 2>/dev/null |
        awk '/^[^ ]/            { sub(/:$/, ""); name = $0 }
             /^ +Candidate:/    { if($2 != "(none)") print name }' ;;
    pacman)
      local p
      # -Sg too: base-devel is a group, which -Si does not know about
      for p in "$@"; do
        if pacman -Si -- "$p" >/dev/null 2>&1 || pacman -Sg -- "$p" >/dev/null 2>&1; then
          printf '%s\n' "$p"
        fi
      done ;;
  esac
  return 0
}

# --skip-unavailable is dnf5, and dnf4 only from 4.20; RHEL 9 ships 4.14 and
# rejects it outright, where --setopt=strict=0 is the older spelling.
# not `dnf ... | grep -q`: grep leaves on the first match and SIGPIPEs dnf,
# and under pipefail the pipeline then reports dnf's status, not the match
_dnf_skip_flag() {
  case "$(dnf install --help 2>&1 || true)" in
    *--skip-unavailable*) printf '%s\n' '--skip-unavailable' ;;
    *)                    printf '%s\n' '--setopt=strict=0' ;;
  esac
}

install_deps() {
  [ -r /etc/os-release ] || die "cannot identify this distribution"
  . /etc/os-release
  local mgr="" pkgs="" family="${ID_LIKE:-$ID}"

  case " $ID $family " in
    *" debian "*|*" ubuntu "*) mgr=apt;    pkgs="$DEB_PACKAGES" ;;
    *" fedora "*|*" rhel "*)   mgr=dnf;    pkgs="$RPM_PACKAGES" ;;
    *" arch "*)                mgr=pacman; pkgs="$ARCH_PACKAGES" ;;
    *) die "unsupported distribution '$ID'; install the build dependencies yourself and re-run with --skip-deps" ;;
  esac

  if [ "$mgr" != apt ]; then
    note "note: the $mgr package list is a best-effort mapping and is not tested by darktable's CI"
  fi

  ask "install build dependencies with $mgr?" || die "cannot build without them (use --skip-deps if they are already installed)"

  # not `[ x ] || cmd`: the command after the last || is not exempt from set -e,
  # so an unreachable repository would end the run with nothing printed
  if [ "$mgr" = apt ]; then
    sudo_ apt-get update ||
      die "apt-get update failed; fix the repository it named and re-run"
  fi
  # pacman is not synced here on purpose: -Sy followed by installing a few
  # packages is the partial-upgrade trap, and -Syu is not this script's call to
  # make. an unsynced database would otherwise filter every name away and look
  # like darktable is unpackageable on Arch
  [ "$mgr" != pacman ] || pacman -Sl >/dev/null 2>&1 ||
    die "pacman has no synced package database; run 'sudo pacman -Syu' first"

  # dnf has a flag for this; apt and pacman need the list narrowed by hand
  if [ "$mgr" != dnf ]; then
    local avail p skipped=""
    # shellcheck disable=SC2086
    avail=" $(_available_packages "$mgr" $pkgs | tr '\n' ' ')"
    for p in $pkgs; do
      case "$avail" in *" $p "*) ;; *) skipped="$skipped $p" ;; esac
    done
    [ -z "$skipped" ] || note "not packaged here, skipping:$skipped"
    pkgs="$avail"
    [ -n "${pkgs// /}" ] || die "$mgr carries none of the build dependencies; is its package list empty?"
  fi

  case "$mgr" in
    apt)    # shellcheck disable=SC2086
            sudo_ apt-get install -y $pkgs ;;
    dnf)    # shellcheck disable=SC2086
            sudo_ dnf install -y "$(_dnf_skip_flag)" $pkgs ;;
    pacman) # shellcheck disable=SC2086
            sudo_ pacman -S --needed --noconfirm $pkgs ;;
  esac
}

# lensfun ships whatever lens database was current when the distribution
# packaged it, which is usually well behind. as root this updates the system
# database, so every user of the machine gets it; a --user install has no root
# and lensfun-update-data falls back to a per-user copy under the data dir
update_lensfun() {
  [ "$SKIP_LENSFUN" = 1 ] && return 0
  command -v lensfun-update-data >/dev/null || {
    note "lensfun-update-data not found, skipping the lens database update"
    return 0
  }
  note "updating the lensfun lens database"
  # it exits 1 both when the database is already current and when it cannot be
  # reached, so the two cannot be told apart: do not call either one a failure
  sudo_ lensfun-update-data ||
    note "no newer lens database, or it could not be fetched; carrying on"
}

# --- desktop integration --------------------------------------------------

# link src to dst. a symlink already pointing into our prefix is ours to
# replace; anything else at that path belongs to something else, very likely a
# package manager, and is kept as .bak rather than destroyed. a second .bak
# would bury the first, so when one is taken nothing is touched at all.
# non-zero says nothing was linked
_link_aside() {
  local src="$1" dst="$2" target=""
  # _canon, not readlink -f: a link into a prefix a failed build has already
  # emptied still has to resolve, or it looks foreign and fills the .bak slot
  [ ! -L "$dst" ] || target="$(_canon "$dst")"
  case "$target" in
    "$PREFIX"/*) ;;
    *) if [ -e "$dst" ] || [ -L "$dst" ]; then
         if [ -e "$dst.bak" ] || [ -L "$dst.bak" ]; then
           note "warning: $dst belongs to something else and $dst.bak is taken, leaving it alone"
           return 1
         fi
         note "$dst belongs to something else, keeping it as $(basename "$dst").bak"
         sudo_ mv -f -- "$dst" "$dst.bak"
       fi ;;
  esac
  sudo_ ln -sfn "$src" "$dst" || { note "warning: could not link $dst"; return 1; }
}

install_desktop() {
  local src="$PREFIX/share/applications/org.darktable.darktable.desktop"
  [ -f "$src" ] || { note "no desktop file in $PREFIX, skipping menu entry"; return 0; }

  # Exec and TryExec are already absolute, so the file only has to be where
  # the menu looks. symlink so the next build is picked up automatically.
  # a distro darktable package owns exactly this path (data/CMakeLists.txt:91)
  local dst="$DATADIR/applications/$(basename "$src")"
  sudo_ mkdir -p "$DATADIR/applications"
  local entry=0
  _link_aside "$src" "$dst" && entry=1 || true

  # Icon=darktable is a bare theme name: it only resolves if the icon sits in
  # a theme directory the system scans
  local icons=0 rel icon
  while IFS= read -r icon; do
    rel="${icon#"$PREFIX"/share/icons/}"
    sudo_ mkdir -p "$DATADIR/icons/$(dirname "$rel")"
    if _link_aside "$icon" "$DATADIR/icons/$rel"; then
      icons=$((icons + 1))
    fi
  done < <(find "$PREFIX/share/icons" -name 'darktable*' -type f 2>/dev/null)

  _refresh_caches "$DATADIR"

  if [ "$entry" = 1 ]; then
    note "linked the desktop entry and $icons icons"
  else
    note "linked $icons icons; the menu entry was left as it was"
  fi
}

# non-zero when darktable itself could not be linked: the install in the prefix
# is fine, but it is not on anyone's PATH, which is not a success
link_binaries() {
  sudo_ mkdir -p "$LINKDIR"
  local linked=0 main_linked=0 name link exe
  for exe in "$PREFIX"/bin/*; do
    [ -x "$exe" ] || continue
    name="$(basename "$exe")"; link="$LINKDIR/$name"
    if _link_aside "$exe" "$link"; then
      linked=$((linked + 1))
      [ "$name" != darktable ] || main_linked=1
    fi
  done
  note "linked $linked binaries into $LINKDIR"

  local found
  found="$(command -v darktable 2>/dev/null || true)"
  [ -n "$found" ] && [ "$found" != "$LINKDIR/darktable" ] &&
    note "warning: $found comes first on your PATH"

  if [ -x "$PREFIX/bin/darktable" ] && [ "$main_linked" = 0 ]; then
    # the usual cause is another prefix installed into the same linkdir, whose
    # links look foreign here and have taken the .bak slots
    note "clear $LINKDIR/darktable and its .bak by hand, or uninstall what else is linked there, and re-run"
    return 1
  fi
  return 0
}

# --- install --------------------------------------------------------------

# build.sh --clean-install only removes what this build tree's manifest lists,
# so a release installed from a different tree would survive underneath the new
# one. empty the prefix, but only once it is clearly a darktable install
clean_prefix() {
  [ -e "$PREFIX" ] || return 0        # main vetted the prefix before building
  # the gate also passes an empty directory, which people pre-create so the
  # install needs no root: nothing to replace there, and nothing to ask about
  _looks_like_darktable "$PREFIX" || return 0
  ask "replace the existing install in $PREFIX?" || die "nothing done"
  note "removing the previous install"
  sudo_ rm -rf -- "$PREFIX"
}

# the header comment is the long help, but piped from curl $0 is "bash" and
# there is no file to read it out of, so fall back to a synopsis
usage() {
  # piped from curl $0 is "bash", with no file to read the header out of
  if [ -r "$0" ]; then
    # the range stops at the first line that is not a comment rather than at a
    # hardcoded number, which inserting a header line would silently truncate
    sed -n '2,${/^#/!q; s/^# \?//; p;}' -- "$0"
    return 0
  fi
  # capitalized to match the header comment this stands in for
  cat <<'EOF'
Build and install an official darktable release on Linux.

Usage: install_release.sh [tag] [options]      (default: the latest release)

Piped from curl there is no file to read the help out of. Save the script and
run "./install_release.sh --help", or read the comment at the top of
https://github.com/darktable-org/darktable/blob/master/tools/install_release.sh
EOF
}

main() {
  local why
  while [ $# -gt 0 ]; do
    case "$1" in
      --list)         ACTION=list ;;
      --clean)        ACTION=clean ;;
      --uninstall)    ACTION=uninstall ;;
      --desktop-only) ACTION=desktop ;;
      --user)         USER_MODE=1; EXPLICIT_TARGET=1 ;;
      --keep-source)  KEEP_SRC=1 ;;
      --with-ai)      WITH_AI=1 ;;
      --skip-deps)    SKIP_DEPS=1 ;;
      --skip-lensfun) SKIP_LENSFUN=1 ;;
      --yes|-y)       ASSUME_YES=1 ;;
      --prefix)       [ $# -ge 2 ] && [ -n "$2" ] || die "--prefix needs a directory"
                      # absolute, so that " ", "." and ".." cannot resolve to
                      # the working directory and have it removed
                      case "$2" in /*) ;;
                        *) die "--prefix needs an absolute path, not '$2'" ;; esac
                      PREFIX="$2"; EXPLICIT_TARGET=1; shift ;;
      # build.sh's, but they undo decisions this script makes on purpose:
      # --install would install before clean_prefix has emptied the prefix, and
      # --sudo would make a --user install root-owned and unremovable
      --install|--sudo)
                      die "$1 is build.sh's; this script decides when to install and when to use root" ;;
      --help|-h)      usage; exit 0 ;;
      # everything past -- is build.sh's, to hand on to cmake
      --)             PASSTHROUGH+=("$@"); break ;;
      # build.sh's value-taking options (build.sh:78-95), and only these. the
      # value travels with the flag, so a bare word left over is unambiguously
      # the release tag: guessing from the shape of the word instead both
      # steals --build-dir 5.6 and misses tags like nightly
      --build-type|--buildtype|--build-dir|--build-generator|-j|--jobs)
                      PASSTHROUGH+=("$1")
                      [ $# -lt 2 ] || { PASSTHROUGH+=("$2"); shift; } ;;
      -*)             PASSTHROUGH+=("$1") ;;
      *)              [ -n "$1" ] || die "empty argument; give a release tag or nothing"
                      [ -z "$TAG" ] || die "more than one release tag given: $TAG and $1"
                      TAG="$1" ;;
    esac
    shift
  done

  # --list before any of the setup below: asking what releases exist is worth
  # answering on a machine that could never build one
  [ "$ACTION" != list ] || { list_releases; exit 0; }

  [ -n "${HOME:-}" ] || die "HOME is not set; this script needs it"
  # the option, not the binary: BusyBox and BSD realpath have no -m, and would
  # otherwise fail later with a bare getopt message and no error: prefix
  realpath -m -- / >/dev/null 2>&1 ||
    die "realpath is missing or has no -m (GNU coreutils provides it)"
  # canonical once, here. the prefix guard compares against $HOME, and a
  # trailing slash or a symlinked component (/home -> /export/home is an
  # ordinary layout) would otherwise walk straight past that comparison
  HOME="$(_canon "$HOME")"
  # after HOME: a relative one would leave SCRATCH relative, and the cleanup
  # trap cannot find the tree again once the build has cd'd into it
  CACHE="${XDG_CACHE_HOME:-$HOME/.cache}"

  # a self-contained prefix in both modes: it can be deleted in one go, which
  # merging into /usr/local or ~/.local would not allow
  if [ "$USER_MODE" = 1 ]; then
    : "${PREFIX:=$HOME/.local/darktable}"
    LINKDIR="${DT_LINKDIR:-$HOME/.local/bin}"
    DATADIR="${XDG_DATA_HOME:-$HOME/.local/share}"
  else
    : "${PREFIX:=/opt/darktable}"
    LINKDIR="${DT_LINKDIR:-/usr/local/bin}"
    DATADIR="/usr/share"
  fi

  # dispatched after the whole command line is read, so options such as --yes
  # apply whichever action was asked for. uninstall runs before the prefix is
  # canonicalized: it wants the name as given, so that a prefix which is itself
  # a symlink has the link removed along with the tree
  case "$ACTION" in
    clean)     clean_scratch; exit 0 ;;
    uninstall) uninstall; exit 0 ;;
  esac

  # canonical from here on: the gate compares against it, and _link_aside
  # matches it against the canonical target of a symlink it may replace.
  # everything below writes into the prefix or links out of it, so this is the
  # one gate both the install and --desktop-only need
  PREFIX="$(_canon "$PREFIX")"
  why="$(_prefix_objection "$PREFIX")" || die "refusing to use $PREFIX: $why"

  if [ "$ACTION" = desktop ]; then
    [ -d "$PREFIX" ] || die "nothing installed at $PREFIX"
    install_desktop
    link_binaries ||
      die "$LINKDIR/darktable could not be linked: see the warning above"
    exit 0
  fi

  for tool in git curl cmake; do
    command -v "$tool" >/dev/null || die "$tool is not installed"
  done

  if [ -z "$TAG" ]; then
    note "asking GitHub for the latest release"
    TAG="$(latest_tag)"
  fi
  note "building $TAG into $PREFIX"
  [ "$USER_MODE" = 1 ] && note "user install: nothing here needs root except the dependencies"

  [ "$SKIP_DEPS" = 1 ] || install_deps

  if [ -z "$SRC" ]; then
    if [ "$KEEP_SRC" = 1 ]; then
      SRC="$HOME/src/darktable-release"
    else
      mkdir -p "$CACHE"
      SCRATCH="$(mktemp -d "$CACHE/darktable-install.XXXXXX")"
      SRC="$SCRATCH/src"
      note "using a scratch tree, removed once installed (--keep-source to keep it)"
    fi
  fi

  if [ -d "$SRC/.git" ]; then
    note "updating $SRC"
    git -C "$SRC" fetch --tags --prune origin
  else
    note "cloning into $SRC"
    mkdir -p "$(dirname "$SRC")"
    # blobless: every tag and commit is there, so switching releases later
    # needs no second clone, but file contents are fetched only for what is
    # checked out. a full clone of darktable is gigabytes, this is ~50M
    git clone --filter=blob:none "$REPO" "$SRC"
  fi

  git -C "$SRC" rev-parse -q --verify "refs/tags/$TAG" >/dev/null ||
    die "no such tag: $TAG (try --list)"
  git -C "$SRC" checkout --quiet --detach "$TAG"
  note "submodules"
  # src/tests/integration is the reference images for the integration test
  # suite: 1.2G, dwarfing darktable itself, and nothing the build reads. take
  # every other submodule by name rather than excluding it after the fact, so
  # one the build gains later is still picked up
  local subs=() sub
  while IFS= read -r sub; do
    [ "$sub" = src/tests/integration ] || subs+=("$sub")
  done < <(git -C "$SRC" config -f .gitmodules --get-regexp '^submodule\..*\.path$' |
             cut -d' ' -f2-)
  [ "${#subs[@]}" -gt 0 ] || die "no submodules listed in .gitmodules"
  git -C "$SRC" submodule update --init --recursive -- "${subs[@]}"

  cd "$SRC"
  local ai=()
  # left to autodetection darktable would build without AI if ONNX Runtime is
  # missing; asked for explicitly, cmake stops instead of quietly omitting it
  [ "$WITH_AI" = 1 ] && ai=(--enable-ai)

  # compile before touching the prefix, so a failed build leaves whatever is
  # already installed there working
  note "compiling"
  ./build.sh "${ai[@]}" --prefix "$PREFIX" "${PASSTHROUGH[@]}"

  clean_prefix

  note "installing"
  # --sudo would make a user install root-owned, and the next --uninstall
  # would then fail to remove it
  local elevate=(--sudo)
  [ "$USER_MODE" = 1 ] && elevate=()
  # clean_prefix has already emptied the prefix by now, so a failure here
  # leaves the previous install's symlinks pointing at nothing. name the one
  # command that clears them, since --uninstall alone will not find a prefix
  # that is no longer there
  local recover="--uninstall --prefix $PREFIX"
  [ "$USER_MODE" = 0 ] || recover="$recover --user"
  ./build.sh "${ai[@]}" --prefix "$PREFIX" --install "${elevate[@]}" "${PASSTHROUGH[@]}" ||
    die "the install failed; run '$0 $recover' to clear what the previous one left behind"

  [ -x "$PREFIX/bin/darktable" ] ||
    die "build finished but $PREFIX/bin/darktable is missing; run '$0 $recover' to clean up"
  local unlinked=0
  link_binaries || unlinked=1
  install_desktop
  update_lensfun

  note "installed: $("$PREFIX/bin/darktable" --version | head -1)"
  # exit non-zero: an install nothing on PATH can find is not what was asked
  # for, and the warning above scrolls past on a build this long
  [ "$unlinked" = 0 ] ||
    die "$PREFIX is installed, but $LINKDIR/darktable could not be linked: see the warning above"
}

# called on the last line so a truncated download - this script is meant to be
# safe to pipe from curl - defines functions and does nothing else
main "$@"
