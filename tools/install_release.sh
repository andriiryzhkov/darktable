#!/usr/bin/env bash
#
# Build and install an official darktable release on Linux.
#
#   ./install_release.sh                  # latest release into /opt/darktable
#   ./install_release.sh release-5.6.0    # a specific tag
#   ./install_release.sh --list           # what releases are available
#   ./install_release.sh --help           # every option
#
# Or run it straight from GitHub without downloading it first:
#
#   curl -fsSL https://raw.githubusercontent.com/andriiryzhkov/darktable/refs/heads/install_tools/tools/install_release.sh | bash
#
# Options go after "bash -s --", which carries them past bash to the script,
# e.g. "| bash -s -- --user".
#
# It still asks before installing packages or replacing an install, because it
# prompts on /dev/tty rather than stdin, which is the script itself when piped.
# Add --yes for an unattended run.
#
# BUILD OPTIONS
#
# Anything this script does not recognize is handed to build.sh, and everything
# after -- goes on to cmake, so --enable-ai, --disable-opencl, -j and the rest
# work; "build.sh --help" lists them. build.sh's own --install and --sudo are
# refused: this script decides when to install and when to use root.
#
# Left alone, cmake enables whatever it autodetects, so a missing dependency
# quietly drops the feature. Asking for one explicitly makes cmake stop instead.
#
# Features in DEFAULT_FEATURES are the exception, for now only AI: cmake
# leaves it off unless asked, so this script asks, as darktable's CI and its
# AppImage do. If the configure step then fails, the build is retried once
# without them and says so; for AI that is when ONNX Runtime or libarchive can
# be neither found nor fetched. --disable-ai leaves it out from the start, and
# --enable-ai makes a missing dependency an error instead.
#
# WHY BUILD FROM SOURCE
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
# The source is the release's own tarball, darktable-<version>.tar.xz, the one
# distributions build from: about 8 MB, with every submodule the build needs and
# without the integration test images. Its SHA-256 is checked against the digest
# GitHub publishes for it; releases before 5.2.0 have none, and say so. A tree
# without .git is also built as a source package (CMakeLists.txt), so a
# compiler newer than darktable's CI warns rather than stops the build.
#
# By default it is unpacked into a scratch directory under ~/.cache and removed
# once the install succeeds. A build wants around 800 MB, and ~/.cache rather
# than /tmp because /tmp is often tmpfs. A failed build keeps its tree and
# prints the path, so you can see what went wrong; --clean removes those
# afterwards. --keep-source keeps it at ~/src/darktable-<version> instead, and
# builds the same release again there without unpacking it again, recompiling
# only what changed. An unpack cut short is noticed and done again.
#
# DEPENDENCIES
#
# curl, tar, xz, sha256sum, cmake and realpath have to be there already.
# Everything the build itself needs is installed for you, unless --skip-deps.
#
# The Debian and Ubuntu list is the one darktable's own CI installs, from
# .github/workflows/ci.yml, plus liblensfun-bin for the lens database update.
# The Fedora and Arch lists are a best-effort mapping and are NOT covered by CI:
# a name a distribution does not carry is skipped rather than failing the run,
# but a genuinely missing dependency then surfaces later as a cmake error.
# Corrections welcome. Any other distribution: install them yourself and use
# --skip-deps.
#
# ONNX Runtime, for AI, is packaged by Ubuntu from 26.04, by Fedora and by Arch.
# Where it is not, darktable's cmake downloads Microsoft's build from GitHub.
#
# REMOVING IT
#
# --uninstall offers whichever installs it finds in /opt/darktable and
# ~/.local/darktable, unless --user or --prefix names one. It also works when
# the prefix is already gone, clearing the symlinks a failed install left
# pointing at nothing - that case has no prefix to find, so it needs --prefix,
# and a failed run prints the exact command. Note that --prefix on its own only
# moves the prefix: the symlinks and the desktop entry still go to the
# system-wide locations unless --user comes with it.
#
# It refuses a prefix that is neither empty nor a darktable, and any directory
# --prefix itself rejects. An empty one that passes those rules is still
# removed, so --prefix is not a safe place to point at something you want kept.
#
# Only symlinks that resolve into the prefix are removed. A file or symlink
# moved aside as .bak at install time is put back, unless something has since
# put a real file there - a distribution package upgrade does - in which case
# the newer file wins. Your settings and library in ~/.config/darktable are
# never touched.

set -euo pipefail

# every assumption below - the package managers, /usr/share, the desktop and
# icon caches - is a Linux one. macOS has its own packaging and Windows builds
# through MSYS2, so fail here rather than half way through
case "$(uname -s)" in
  Linux) ;;
  Darwin) printf 'this script is Linux only; on macOS use the .dmg or homebrew\n' >&2; exit 1 ;;
  *)      printf 'this script is Linux only (found %s)\n' "$(uname -s)" >&2; exit 1 ;;
esac

API="https://api.github.com/repos/darktable-org/darktable/releases"
DOWNLOAD="https://github.com/darktable-org/darktable/releases/download"
# DT_SRC names a tree to keep and reuse. with neither it nor --keep-source the
# source is unpacked into a scratch directory and removed once installed.
# under $HOME rather than /tmp: a build wants ~800M, and /tmp is frequently
# tmpfs, so this would otherwise be ~800M of RAM
SRC="${DT_SRC:-}"
# in a tree this script unpacked: the release it holds, see fetch_source
SOURCE_MARK=".install_release"
CACHE=""    # set in main, once HOME is canonical
SCRATCH=""
PREFIX="" LINKDIR="" DATADIR=""
USER_MODE=0
SKIP_DEPS=0 SKIP_LENSFUN=0 ASSUME_YES=0 KEEP_SRC=0
ACTION=install
EXPLICIT_TARGET=0
TAG="" PASSTHROUGH=()
# build.sh features turned on unless the command line decides either way,
# for what cmake leaves off but darktable's own builds turn on: AI is on in
# its CI (.ci/ci-script.sh) and its AppImage (tools/appimage-build-script.sh)
DEFAULT_FEATURES="ai"

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
  libarchive-devel onnxruntime-devel
"

ARCH_PACKAGES="
  base-devel cmake ninja git gettext intltool
  desktop-file-utils appstream-glib perl po4a
  cairo colord colord-gtk cups exiv2 gtk3 gmic graphicsmagick libraw lcms2
  lensfun libavif curl libgphoto2 libheif libjpeg-turbo libpng librsvg
  libsecret libtiff libwebp libxml2 libxslt json-glib lua opencv openexr
  openjpeg2 osm-gps-map portmidi potrace pugixml sdl2 sqlite zlib cmocka
  python-jsonschema onnxruntime-cpu
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
  # and a download that was interrupted before fetch_source could remove it
  for d in "$CACHE"/darktable-install.* "$CACHE"/darktable-source.*; do
    [ -e "$d" ] || continue          # no match leaves the glob unexpanded
    printf '  %s  %s\n' "$(du -sh "$d" 2>/dev/null | cut -f1)" "$d"
    found=$((found + 1))
  done

  if [ "$found" = 0 ]; then
    note "no leftover source trees in $CACHE"
  else
    ask "remove these $found leftovers?" || die "nothing done"
    rm -rf -- "$CACHE"/darktable-install.* "$CACHE"/darktable-source.*
    note "removed $found"
  fi

  # the kept trees are deliberate, so say they are there rather than delete
  # them. darktable-release is where versions before the tarball kept theirs
  for d in "$HOME"/src/darktable-[0-9]* "$HOME/src/darktable-release"; do
    [ -d "$d" ] || continue
    note "note: the --keep-source tree at $d ($(du -sh "$d" 2>/dev/null | cut -f1)) is left alone"
  done
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

# unpack the release's own source tarball into $SRC, an empty directory,
# checked against the digest GitHub publishes for it when there is one.
# SOURCE_MARK says "unpacking" until the last file is out, then names the
# release, so a run cut short is not mistaken for a tree to reuse
fetch_source() {
  local name="darktable-${TAG#release-}.tar.xz" json digest tarball
  # curl -f fails alike on a missing release, a rate limit and no network
  json="$(curl -fsSL "$API/tags/$TAG")" ||
    die "could not look up $TAG on GitHub: no such release (try --list), or GitHub is unreachable or rate-limited"
  # the quoted name: the .asc beside it and the download URL contain it too
  printf '%s\n' "$json" | grep '"name"' | grep -qF "\"$name\"" ||
    die "release $TAG has no $name to build from"
  # the asset's digest follows its name; its download URL closes it. one
  # uploaded before GitHub kept digests has "digest": null
  digest="$(printf '%s\n' "$json" | awk -v q="\"$name\"" '
    /"name"/ && index($0, q) { found = 1 }
    found && /"digest"/ {
      if (match($0, /sha256:[0-9a-f]+/)) print substr($0, RSTART + 7, RLENGTH - 7)
      exit
    }
    found && /"browser_download_url"/ { exit }')"

  # in the scratch tree when there is one, which an interrupted run keeps for
  # --clean anyway. otherwise --clean knows this name
  mkdir -p "$CACHE"
  tarball="$(mktemp "${SCRATCH:-$CACHE}/darktable-source.XXXXXX")"
  note "downloading $name"
  curl -fL --progress-bar -o "$tarball" "$DOWNLOAD/$TAG/$name" ||
    { rm -f "$tarball"; die "could not download $name"; }
  if [ -n "$digest" ]; then
    printf '%s  %s\n' "$digest" "$tarball" | sha256sum -c --quiet - ||
      { rm -f "$tarball"; die "$name does not match the digest GitHub publishes for it"; }
    note "checksum verified"
  else
    note "warning: GitHub publishes no digest for $name, so it is not verified"
  fi
  mkdir -p "$SRC"
  printf 'unpacking %s\n' "$TAG" > "$SRC/$SOURCE_MARK"
  tar -xJf "$tarball" -C "$SRC" --strip-components=1 ||
    { rm -f "$tarball"; die "could not unpack $name"; }
  printf '%s\n' "$TAG" > "$SRC/$SOURCE_MARK"
  rm -f "$tarball"
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

# a here-doc rather than the header comment: piped from curl $0 is "bash" and
# stdin is spent, so there is nothing to read the options back out of
usage() {
  cat <<EOF
install_release.sh [tag] [options] [-- [additional cmake configuration options...]]

Build and install an official darktable release. With no tag, the latest one.

Options:
Installation:
   --prefix         <string>  Install directory prefix, absolute, and either
                              empty or already a darktable
                              (default: /opt/darktable)
   --user                     Install under \$HOME, needing no root
                              (prefix: ~/.local/darktable)
   --yes                      Do not ask before installing packages or
                              replacing an install

Build:
   --skip-deps                Do not touch the package manager
   --skip-lensfun             Do not update the lensfun lens database
   --keep-source              Keep the source tree and reuse it next time
                              (default: ~/src/darktable-<version>)

Actual actions:
   --list                     Print the recent releases and exit
   --desktop-only             Only refresh the menu entry and icons
   --uninstall                Remove an install made by this script
   --clean                    Remove source trees left by failed builds

Additional build.sh and cmake options:
Anything not listed above is passed to build.sh, so --disable-ai,
--disable-opencl and -j work; "build.sh --help" lists them. AI is built
by default, and without it if it cannot be configured; --disable-ai
leaves it out, --enable-ai makes it required. build.sh's --install
and --sudo are refused.

Environment:
   DT_SRC                     Source tree to keep and reuse: empty, or
                              one this script unpacked
   DT_LINKDIR                 Where the symlinks go
                              (default: /usr/local/bin, ~/.local/bin with --user)

Extra:
-h --help                     Print help message

The comment at the top of this script has the rest: build options, why build
from source, where it installs, the source tree, dependencies, and removing it.
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
  # a default feature is added unless the line already decides it, either way
  # and wherever: as build.sh's switch or as the cmake option it stands for
  local feature arg decided defaulted=()
  for feature in $DEFAULT_FEATURES; do
    decided=0
    for arg in "${PASSTHROUGH[@]}"; do
      case "$arg" in
        --enable-"$feature"|--disable-"$feature"|-DUSE_"${feature^^}"=*) decided=1 ;;
      esac
    done
    [ "$decided" = 1 ] || defaulted+=("$feature")
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

  for tool in curl tar xz sha256sum cmake; do
    command -v "$tool" >/dev/null || die "$tool is not installed"
  done

  if [ -z "$TAG" ]; then
    note "asking GitHub for the latest release"
    TAG="$(latest_tag)"
  fi
  note "building $TAG into $PREFIX"
  [ "$USER_MODE" = 1 ] && note "user install: nothing here needs root except the dependencies"

  [ "$SKIP_DEPS" = 1 ] || install_deps

  local version="${TAG#release-}"
  if [ -z "$SRC" ]; then
    if [ "$KEEP_SRC" = 1 ]; then
      SRC="$HOME/src/darktable-$version"
    else
      mkdir -p "$CACHE"
      SCRATCH="$(mktemp -d "$CACHE/darktable-install.XXXXXX")"
      SRC="$SCRATCH/src"
      note "using a scratch tree, removed once installed (--keep-source to keep it)"
    fi
  fi

  # a tree marked with this release is built again in place. another mark is
  # still ours, since fetch_source only unpacks into an empty directory: an
  # unpack cut short, or a kept tree of another release, cleared and redone.
  # anything unmarked is not ours to overwrite
  local mark=""
  [ -f "$SRC/$SOURCE_MARK" ] && mark="$(cat "$SRC/$SOURCE_MARK")"
  if [ "$mark" = "$TAG" ]; then
    note "reusing the $version source in $SRC"
  elif [ -n "$mark" ]; then
    note "clearing $SRC ($mark), to unpack $version"
    find "$SRC" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
    fetch_source
  elif [ -n "$(ls -A "$SRC" 2>/dev/null)" ]; then
    die "$SRC holds something other than the $version source; remove it, or give DT_SRC an empty directory"
  else
    fetch_source
  fi

  cd "$SRC"
  # compile before touching the prefix, so a failed build leaves whatever is
  # already installed there working
  note "compiling"
  # the switches go first: PASSTHROUGH may hold a "--", and build.sh hands
  # whatever follows it to cmake untouched
  local build_args=() dropped=0
  for feature in "${defaulted[@]}"; do build_args+=(--enable-"$feature"); done
  build_args+=("${PASSTHROUGH[@]}")
  # logged to tell a failed configure step from a failed compile: only the
  # first can be down to a feature this script asked for, and a compile error
  # must not be retried without it and hidden
  local log="$SRC/install_release.log"
  if ! ./build.sh --prefix "$PREFIX" "${build_args[@]}" 2>&1 | tee "$log"; then
    [ "${#defaulted[@]}" -gt 0 ] &&
      grep -q 'Configuring incomplete, errors occurred' "$log" ||
      die "the build failed: see above"
    note "warning: configuring failed, see above; building again without ${defaulted[*]}"
    build_args=()
    for feature in "${defaulted[@]}"; do build_args+=(--disable-"$feature"); done
    build_args+=("${PASSTHROUGH[@]}")
    dropped=1
    ./build.sh --prefix "$PREFIX" "${build_args[@]}"
  fi

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
  # the same switches as the build, or AI would be configured back in
  ./build.sh --prefix "$PREFIX" --install "${elevate[@]}" "${build_args[@]}" ||
    die "the install failed; run '$0 $recover' to clear what the previous one left behind"

  [ -x "$PREFIX/bin/darktable" ] ||
    die "build finished but $PREFIX/bin/darktable is missing; run '$0 $recover' to clean up"
  local unlinked=0
  link_binaries || unlinked=1
  install_desktop
  update_lensfun

  note "installed: $("$PREFIX/bin/darktable" --version | head -1)"
  # again here: the warning came before the long compile and scrolled past
  [ "$dropped" = 0 ] ||
    note "warning: built without ${defaulted[*]}, which could not be configured; install what it needs and re-run to get it"
  # exit non-zero: an install nothing on PATH can find is not what was asked
  # for, and the warning above scrolls past on a build this long
  [ "$unlinked" = 0 ] ||
    die "$PREFIX is installed, but $LINKDIR/darktable could not be linked: see the warning above"
}

# called on the last line so a truncated download - this script is meant to be
# safe to pipe from curl - defines functions and does nothing else
main "$@"
