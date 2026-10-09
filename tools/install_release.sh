#!/usr/bin/env bash
#
# Build and install an official darktable release on Linux.
#
#   ./install_release.sh                  # latest release into /opt/darktable
#   ./install_release.sh release-5.6.0    # a specific tag
#   ./install_release.sh --user           # into ~/.local/darktable, root only for deps
#   ./install_release.sh --help           # every option
#
# Or straight from GitHub, with options after "bash -s --":
#
#   curl -fsSL https://raw.githubusercontent.com/andriiryzhkov/darktable/refs/heads/install_tools/tools/install_release.sh | bash
#
# It installs the build dependencies, downloads the release tarball and checks
# it against the digest GitHub publishes, and builds it with AI on, as
# darktable's CI and AppImage do, and for this CPU (-march=native, see
# cmake/march-mtune.cmake). The install is a directory of its own,
# /opt/darktable or ~/.local/darktable, linked into /usr/local or ~/.local, so
# that --uninstall can remove all of it. Settings and the library in
# ~/.config/darktable are never touched.

set -euo pipefail

case "$(uname -s)" in
  Linux) ;;
  *) printf 'this script is Linux only (found %s)\n' "$(uname -s)" >&2; exit 1 ;;
esac

API="https://api.github.com/repos/darktable-org/darktable/releases"
DOWNLOAD="https://github.com/darktable-org/darktable/releases/download"
# build.sh features turned on unless the command line decides either way: cmake
# leaves AI off, but darktable's CI (.ci/ci-script.sh) and AppImage
# (tools/appimage-build-script.sh) turn it on
DEFAULT_FEATURES="ai"

USER_MODE=0 SKIP_DEPS=0 SKIP_LENSFUN=0 ASSUME_YES=0 KEEP_SRC=0
ACTION=install TAG="" PASSTHROUGH=()
BUILD_DIR="" BUILD_TYPE_GIVEN=0
PREFIX="" PREFIX_REAL="" LINKDIR="" DATADIR="" SCRATCH="" SRC=""
CMAKE=""       # absolute: sudo's secure_path need not have it

# Debian and Ubuntu: what darktable's CI installs (.github/workflows/ci.yml),
# plus liblensfun-bin for the lens database update. Fedora and Arch are a
# best-effort mapping, not covered by CI. A name the distribution does not
# carry is skipped; ONNX Runtime, where missing, is downloaded by cmake
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
note() { printf '\033[1m==>\033[0m %s\n' "$*"; }

# a failed build keeps its tree, and the log in it, until the next run
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

# on /dev/tty, not stdin: piped from curl, stdin is the script itself
ask() {
  [ "$ASSUME_YES" = 1 ] && return 0
  { : </dev/tty; } 2>/dev/null ||
    die "nothing to prompt on; re-run with --yes to accept: $1"
  local reply=""
  printf '%s [y/N] ' "$1"
  read -r reply </dev/tty || true
  case "$reply" in [yY]*) return 0 ;; *) return 1 ;; esac
}

# root for the install and the links, unless --user or root already
sudo_() {
  if [ "$USER_MODE" = 1 ] || [ "$(id -u)" = 0 ]; then "$@"; else sudo "$@"; fi
}

# root for the package manager, in either mode
_as_root() {
  if [ "$(id -u)" = 0 ]; then "$@"; else sudo "$@"; fi
}

# true when $1 resolves into the prefix: a link of ours, dangling or not
_ours() {
  case "$(realpath -m -- "$1")" in "$PREFIX_REAL"/*) return 0 ;; esac
  return 1
}

# the contents, not the directory: it may be a mount point
_empty_prefix() {
  sudo_ find "$PREFIX/" -mindepth 1 -maxdepth 1 ! -name lost+found -exec rm -rf -- {} +
}

# it gets emptied, so it must be absent, empty or an install (then true).
# not a symlink: the path is ours by name, but whatever a link points at is not
_check_prefix() {
  [ -e "$PREFIX" ] || [ -L "$PREFIX" ] || return 1
  [ ! -L "$PREFIX" ] || die "$PREFIX is a symlink; make it a directory or a mount point"
  [ -d "$PREFIX" ] && [ -r "$PREFIX" ] && [ -x "$PREFIX" ] ||
    die "$PREFIX is not a directory, or not one this user can read"
  [ -n "$(find "$PREFIX/" -mindepth 1 -maxdepth 1 ! -name lost+found -print -quit)" ] ||
    return 1
  # any part of an install: one cut short has only what went first, and
  # cmake installs po, then src, then data (CMakeLists.txt:380,464-465)
  [ -d "$PREFIX/share/locale" ] || [ -d "$PREFIX/share/darktable" ] ||
    [ -e "$PREFIX/bin/darktable" ] || compgen -G "$PREFIX/lib*/darktable" >/dev/null ||
    die "$PREFIX is neither empty nor a darktable install"
}

_refresh_caches() {
  command -v update-desktop-database >/dev/null &&
    sudo_ update-desktop-database -q "$DATADIR/applications" || true
  # only a cache that exists, as xdg-icon-resource does: a new one that nothing
  # else updates would hide icons installed later. -t: the hicolor directory
  # under /usr/local or ~/.local usually has no index.theme
  [ -f "$DATADIR/icons/hicolor/icon-theme.cache" ] &&
    command -v gtk-update-icon-cache >/dev/null &&
    sudo_ gtk-update-icon-cache -q -t -f "$DATADIR/icons/hicolor" || true
}

# --- releases ---------------------------------------------------------------

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

# the release tarball, unpacked beside $SRC and moved into place only once
# complete, so that a $SRC which exists holds the whole tree
fetch_source() {
  local name="darktable-${TAG#release-}.tar.xz" json digest tmp="$SRC.tmp"
  # here-strings, not pipes: grep -q and awk leave early, and under pipefail
  # the writer's SIGPIPE would fail the lookup
  json="$(curl -fsSL "$API/tags/$TAG")" ||
    die "could not look up $TAG on GitHub: no such release (try --list), or GitHub is unreachable or rate-limited"
  grep -q "\"name\": *\"$name\"" <<< "$json" ||
    die "release $TAG has no $name to build from"
  # releases before 5.2.0 have "digest": null
  digest="$(awk -v q="\"$name\"" '
    /"name"/ && index($0, q) { found = 1 }
    found && /"digest"/ {
      if (match($0, /sha256:[0-9a-f]+/)) print substr($0, RSTART + 7, RLENGTH - 7)
      exit
    }
    found && /"browser_download_url"/ { exit }' <<< "$json")"

  rm -rf -- "$tmp"
  mkdir -p "$tmp"
  note "downloading $name"
  curl -fL --progress-bar -o "$tmp.tar.xz" "$DOWNLOAD/$TAG/$name" ||
    die "could not download $name"
  if [ -n "$digest" ]; then
    printf '%s  %s\n' "$digest" "$tmp.tar.xz" | sha256sum -c --quiet - ||
      die "$name does not match the digest GitHub publishes for it"
    note "checksum verified"
  else
    note "warning: GitHub publishes no digest for $name, so it is not verified"
  fi
  tar -xJf "$tmp.tar.xz" -C "$tmp" --strip-components=1 || die "could not unpack $name"
  rm -f -- "$tmp.tar.xz"
  mv -- "$tmp" "$SRC"
}

# --- dependencies -----------------------------------------------------------

# the names this manager actually carries, one per line: apt-get has no
# tolerant mode, and pacman -S drops the whole transaction on one unknown name
_available_packages() {
  local mgr="$1"; shift
  case "$mgr" in
    apt)
      # a name with no candidate is virtual, which apt-get would refuse too.
      # LC_ALL=C: the field names are translated
      LC_ALL=C apt-cache policy -- "$@" 2>/dev/null |
        awk '/^[^ ]/            { sub(/:$/, ""); name = $0 }
             /^ +Candidate:/    { if($2 != "(none)") print name }' ;;
    pacman)
      local p
      # -Sg too: base-devel is a group
      for p in "$@"; do
        if pacman -Si -- "$p" >/dev/null 2>&1 || pacman -Sg -- "$p" >/dev/null 2>&1; then
          printf '%s\n' "$p"
        fi
      done ;;
  esac
  return 0
}

# --skip-unavailable is dnf5, and dnf4 only from 4.20; RHEL 9 ships 4.14
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

  # no is --skip-deps asked late: they may well be installed already, and if
  # not, configuring names what is missing
  if ! ask "install build dependencies with $mgr?"; then
    note "skipping the build dependencies; the build stops if one is missing"
    return 0
  fi

  if [ "$mgr" = apt ]; then
    _as_root apt-get update ||
      die "apt-get update failed; fix the repository it named and re-run"
  fi
  # not synced here: -Sy and then installing a few packages is a partial
  # upgrade, and -Syu is not this script's call to make
  [ "$mgr" != pacman ] || pacman -Sl >/dev/null 2>&1 ||
    die "pacman has no synced package database; run 'sudo pacman -Syu' first"

  # either curl flavor will do, and the two conflict: asking for gnutls where
  # openssl is installed would remove it
  if [ "$mgr" = apt ]; then
    case "$(dpkg-query -W -f='${Status}' libcurl4-openssl-dev 2>/dev/null || true)" in
      *" installed") pkgs="${pkgs/libcurl4-gnutls-dev/libcurl4-openssl-dev}" ;;
    esac
  fi

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
    # --no-remove: apt would otherwise settle a conflict by removing packages
    apt)    # shellcheck disable=SC2086
            _as_root apt-get install -y --no-remove $pkgs ;;
    dnf)    # shellcheck disable=SC2086
            _as_root dnf install -y "$(_dnf_skip_flag)" $pkgs ;;
    pacman) # shellcheck disable=SC2086
            _as_root pacman -S --needed --noconfirm $pkgs ;;
  esac
}

# distributions ship the lens database as it was when they packaged lensfun.
# as root this updates it for every user; without root lensfun-update-data
# keeps a per-user copy
update_lensfun() {
  [ "$SKIP_LENSFUN" = 1 ] && return 0
  command -v lensfun-update-data >/dev/null || {
    note "lensfun-update-data not found, skipping the lens database update"
    return 0
  }
  note "updating the lensfun lens database"
  # it exits 1 both when the database is current and when it cannot be reached
  sudo_ lensfun-update-data ||
    note "no newer lens database, or it could not be fetched; carrying on"
}

# --- links ------------------------------------------------------------------

# link $1 at $2, unless something that is not ours is already there
_link() {
  if { [ -e "$2" ] || [ -L "$2" ]; } && ! { [ -L "$2" ] && _ours "$2"; }; then
    note "warning: $2 is not ours, leaving it alone"
    return 1
  fi
  sudo_ ln -sfn "$1" "$2"
}

# non-zero when darktable itself could not be linked
link_binaries() {
  sudo_ mkdir -p "$LINKDIR"
  local linked=0 main_linked=0 exe
  for exe in "$PREFIX"/bin/*; do
    [ -x "$exe" ] || continue
    if _link "$exe" "$LINKDIR/${exe##*/}"; then
      linked=$((linked + 1))
      [ "${exe##*/}" != darktable ] || main_linked=1
    fi
  done
  note "linked $linked binaries into $LINKDIR"

  local found
  found="$(command -v darktable 2>/dev/null || true)"
  [ -n "$found" ] && [ "$found" != "$LINKDIR/darktable" ] &&
    note "warning: $found comes first on your PATH"
  [ "$main_linked" = 1 ]
}

# Exec in the desktop file is absolute, so a link is all the menu needs, and
# it follows the next install. /usr/local/share is searched ahead of
# /usr/share, where a distribution's darktable puts the same file
install_desktop() {
  local src="$PREFIX/share/applications/org.darktable.darktable.desktop"
  [ -f "$src" ] || { note "no desktop file in $PREFIX, skipping menu entry"; return 0; }
  sudo_ mkdir -p "$DATADIR/applications"
  local entry=0 icons=0 icon rel
  _link "$src" "$DATADIR/applications/${src##*/}" && entry=1
  while IFS= read -r icon; do
    rel="${icon#"$PREFIX"/share/icons/}"
    sudo_ mkdir -p "$DATADIR/icons/${rel%/*}"
    _link "$icon" "$DATADIR/icons/$rel" && icons=$((icons + 1))
  done < <(find "$PREFIX/share/icons" -name 'darktable*' -type f 2>/dev/null)
  _refresh_caches
  if [ "$entry" = 1 ]; then
    note "linked the desktop entry and $icons icons"
  else
    note "linked $icons icons; the menu entry was left as it was"
  fi
}

# also clears what a failed install left, links into a prefix already emptied
uninstall() {
  local links=() link
  for link in "$LINKDIR"/* "$DATADIR"/applications/*darktable* \
              "$DATADIR"/icons/hicolor/*/apps/darktable*; do
    if [ -L "$link" ] && _ours "$link"; then links+=("$link"); fi
  done
  [ -d "$PREFIX" ] || [ "${#links[@]}" -gt 0 ] || die "nothing installed at $PREFIX"
  note "about to remove $PREFIX and ${#links[@]} symlinks into it"
  note "your settings and library in ~/.config/darktable are kept"
  ask "remove it?" || die "nothing done"
  for link in "${links[@]}"; do sudo_ rm -f -- "$link"; done
  if [ -d "$PREFIX" ]; then
    _empty_prefix || die "could not remove everything in $PREFIX: see above"
    sudo_ rmdir -- "$PREFIX" 2>/dev/null || true
  fi
  _refresh_caches
  note "removed"
}

usage() {
  cat <<EOF
Usage: install_release.sh [options] [release tag] [-- cmake options]

Build and install an official darktable release. With no tag, the latest one.

   --user                     Install under \$HOME, needing no root except for
                              the dependencies (~/.local/darktable, linked into
                              ~/.local/bin and ~/.local/share). Without it:
                              /opt/darktable, linked into /usr/local
   -y --yes                   Do not ask before installing packages or
                              replacing an install
   --skip-deps                Do not touch the package manager
   --skip-lensfun             Do not update the lensfun lens database
   --keep-source              Keep the source in ~/src/darktable-<version> and
                              reuse it next time

   --list                     Print the recent releases and exit
   --uninstall                Remove the install (with --user, the user one)
   --desktop-only             Only relink the binaries, menu entry and icons
   -h --help                  Print this help

build.sh's --enable-X, --disable-X, --asan, --build-type, --build-dir,
--build-generator and -j are passed to it, and anything after -- to
cmake; "build.sh --help" lists them. AI is built by default, and
without it if it cannot be configured; --disable-ai leaves it out,
--enable-ai makes it required. The build type is Release unless
--build-type says otherwise.

Environment:
   DT_LINKDIR                 Where the binaries are linked
                              (default: /usr/local/bin, ~/.local/bin with --user)
EOF
}

# $1 is enable or disable, for main's defaulted features. the switches go
# first: PASSTHROUGH may hold a "--", and build.sh hands what follows to cmake
_build() {
  local args=() feature
  [ "$BUILD_TYPE_GIVEN" = 1 ] || args+=(--build-type Release)
  for feature in "${defaulted[@]}"; do args+=(--"$1"-"$feature"); done
  ( cd "$SRC" && ./build.sh --prefix "$PREFIX" "${args[@]}" "${PASSTHROUGH[@]}" ) 2>&1 |
    tee -a "$log"
}

main() {
  # sudo combines its umask with the caller's, so a 077 would leave the install
  # and the system-wide desktop and icon caches unreadable to other users
  umask 022
  local a replace=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --list)         ACTION=list ;;
      --uninstall)    ACTION=uninstall ;;
      --desktop-only) ACTION=desktop ;;
      --user)         USER_MODE=1 ;;
      --keep-source)  KEEP_SRC=1 ;;
      --skip-deps)    SKIP_DEPS=1 ;;
      --skip-lensfun) SKIP_LENSFUN=1 ;;
      --yes|-y)       ASSUME_YES=1 ;;
      --help|-h)      usage; exit 0 ;;
      # build.sh's, but this script decides what to build, when to install
      # and when to use root
      --install|--sudo|--skip-build|--skip-config|\
      --clean-build|--clean-install|--clean-all)
                      die "$1 is build.sh's; this script does that itself" ;;
      # build.sh puts these after its own -DCMAKE_INSTALL_PREFIX, so they win
      # and install where --uninstall does not look
      --)             for a in "$@"; do
                        case "$a" in
                          # anywhere in the word: build.sh evals the lot
                          *CMAKE_INSTALL_PREFIX[:=]*|--install-prefix*)
                            die "the install prefix is not for changing;" \
                                "use --user for a user install" ;;
                          *CMAKE_INSTALL_*DIR[:=]*)
                            die "$a: the install directories are not for changing" ;;
                        esac
                      done
                      PASSTHROUGH+=("$@"); break ;;
      # build.sh hands cmake only what follows -- (build.sh:124-127)
      -D*)            die "$1 is a cmake option; cmake options go after --" ;;
      # the value travels with the flag, so a bare word left over is the tag
      --build-type|--buildtype|--build-dir|--build-generator|-j|--jobs)
                      case "${2:--}" in -*|release-*) die "$1 needs a value" ;; esac
                      [ "$1" != --build-dir ] || BUILD_DIR="$2"
                      case "$1" in --build-type|--buildtype) BUILD_TYPE_GIVEN=1 ;; esac
                      PASSTHROUGH+=("$1" "$2"); shift ;;
      --enable-*|--disable-*|--asan)
                      PASSTHROUGH+=("$1") ;;
      # build.sh would only warn about these and carry on without them
      --build-type=*|--buildtype=*|--build-dir=*|--build-generator=*|--jobs=*)
                      die "write '${1%%=*} ${1#*=}': build.sh takes no --option=value" ;;
      -*)             die "unknown option $1; see --help" ;;
      *)              [ -n "$1" ] || die "empty argument; give a release tag or nothing"
                      [ -z "$TAG" ] || die "more than one release tag given: $TAG and $1"
                      TAG="$1" ;;
    esac
    shift
  done
  if [ "$ACTION" != install ]; then
    [ "${#PASSTHROUGH[@]}" = 0 ] ||
      die "build options do not apply here: ${PASSTHROUGH[*]}"
    [ -z "$TAG" ] || die "a release tag does not apply here"
  fi

  # a default feature is added unless the line already decides it: as
  # build.sh's switch before --, or as the cmake option after it
  local feature arg decided past defaulted=()
  for feature in $DEFAULT_FEATURES; do
    decided=0 past=0
    for arg in "${PASSTHROUGH[@]}"; do
      if [ "$past" = 0 ]; then
        case "$arg" in
          --) past=1 ;;
          --enable-"$feature"|--disable-"$feature") decided=1 ;;
        esac
      else
        case "$arg" in
          -DUSE_"${feature^^}"[:=]*|USE_"${feature^^}"[:=]*) decided=1 ;;
        esac
      fi
    done
    [ "$decided" = 1 ] || defaulted+=("$feature")
  done

  [ "$ACTION" != list ] || { list_releases; exit 0; }

  if [ "$USER_MODE" = 1 ]; then
    PREFIX="$HOME/.local/darktable"
    LINKDIR="${DT_LINKDIR:-$HOME/.local/bin}"
    DATADIR="${XDG_DATA_HOME:-$HOME/.local/share}"
  else
    PREFIX="/opt/darktable"
    LINKDIR="${DT_LINKDIR:-/usr/local/bin}"
    DATADIR="/usr/local/share"
  fi
  # once, and fatal: _ours with an empty one would claim every link
  PREFIX_REAL="$(realpath -m -- "$PREFIX")" && [ -n "$PREFIX_REAL" ] ||
    die "realpath could not resolve $PREFIX"
  _check_prefix && replace=1

  case "$ACTION" in
    uninstall) uninstall; exit 0 ;;
    desktop)   [ -x "$PREFIX/bin/darktable" ] || die "nothing installed at $PREFIX"
               install_desktop
               link_binaries ||
                 die "$LINKDIR/darktable could not be linked: see the warning above"
               exit 0 ;;
  esac

  for tool in curl tar xz sha256sum; do
    command -v "$tool" >/dev/null || die "$tool is not installed"
  done

  if [ -z "$TAG" ]; then
    note "asking GitHub for the latest release"
    TAG="$(latest_tag)"
  fi
  note "building $TAG into $PREFIX"
  # asked now rather than after the build: a long compile is wasted when the
  # answer is no, and keys pressed while it ran could answer it
  if [ "$replace" = 1 ]; then
    ask "replace the existing install in $PREFIX?" || die "nothing done"
  fi
  # _link would refuse it too, but only after the build
  local dt="$LINKDIR/darktable"
  { [ -L "$dt" ] && _ours "$dt"; } || { [ ! -e "$dt" ] && [ ! -L "$dt" ]; } ||
    die "$dt is not a link into $PREFIX (another darktable?); remove it and re-run"

  [ "$SKIP_DEPS" = 1 ] || install_deps
  CMAKE="$(command -v cmake)" || die "cmake is not installed"
  # the install runs it with sudo_, after the prefix is emptied
  sudo_ "$CMAKE" --version >/dev/null ||
    die "$CMAKE does not run for the install: see above"

  local version="${TAG#release-}"
  if [ "$KEEP_SRC" = 1 ]; then
    SRC="$HOME/src/darktable-$version"
  else
    # under ~/.cache, not /tmp: a build wants ~800 MB, and /tmp is often tmpfs
    SCRATCH="${XDG_CACHE_HOME:-$HOME/.cache}/darktable-install"
    rm -rf -- "$SCRATCH"
    SRC="$SCRATCH/src"
  fi
  if [ -f "$SRC/build.sh" ]; then
    note "reusing the $version source in $SRC"
  elif [ -e "$SRC" ] || [ -L "$SRC" ]; then
    die "$SRC is not a whole release tree; remove it and re-run"
  else
    mkdir -p "${SRC%/*}"
    fetch_source
  fi

  note "compiling"
  # logged to tell a failed configure step from a failed compile: only the
  # former is worth retrying without the default features
  local log="$SRC/install_release.log" dropped=0
  : > "$log"
  if ! _build enable; then
    [ "${#defaulted[@]}" -gt 0 ] &&
      grep -q 'Configuring incomplete, errors occurred' "$log" ||
      die "the build failed: see above"
    note "warning: configuring failed, see above;" \
      "retrying without ${defaulted[*]} in case that is the cause"
    dropped=1
    _build disable || die "the build failed, with and without ${defaulted[*]}: see above"
  fi

  # emptied rather than installed over: files an older release had and this
  # one does not, such as a dropped plugin, would otherwise still be loaded
  note "installing"
  local build="${BUILD_DIR:-build}" again="--uninstall" rc=0
  case "$build" in /*) ;; *) build="$SRC/$build" ;; esac
  [ "$USER_MODE" = 0 ] || again="--uninstall --user"
  [ ! -d "$PREFIX" ] || _empty_prefix || die "could not empty $PREFIX: see above"
  # cmake --install, not build.sh --install, which would build again first; a
  # source package rewrites version_gen.c on every build, so everything would
  # be relinked. the manifest goes with the privilege that wrote it, or a
  # root-owned one would break a later --user install from a kept tree
  sudo_ rm -f -- "$build/install_manifest.txt" || true
  { sudo_ mkdir -p "$PREFIX" &&
      sudo_ chmod 755 "$PREFIX" &&
      sudo_ "$CMAKE" --install "$build"; } || rc=$?
  sudo_ rm -f -- "$build/install_manifest.txt" || true
  [ "$rc" = 0 ] || die "the install failed; $again clears what is left of it"
  [ -x "$PREFIX/bin/darktable" ] ||
    die "the install finished but $PREFIX/bin/darktable is missing;" \
      "$again clears what is left"

  local unlinked=0
  link_binaries || unlinked=1
  install_desktop
  update_lensfun

  note "installed: $("$PREFIX/bin/darktable" --version | head -1)"
  [ "$dropped" = 0 ] ||
    note "warning: built without ${defaulted[*]}, which could not be configured; install what it needs and re-run to get it"
  [ "$unlinked" = 0 ] ||
    die "$PREFIX is installed, but $LINKDIR/darktable could not be linked: see the warning above"
}

main "$@"
