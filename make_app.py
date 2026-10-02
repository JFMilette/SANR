"""
Build SANR.app, a macOS application that starts this folder's main.py with
its .venv Python, so the Dock, the app switcher and the menu bar show SANR
and its icon instead of python3.12.

Run with   python make_app.py [destination folder]   (default: this folder)

The app holds the absolute paths of this folder and of its Python: build it
again if either moves.  It can be dragged to the Dock or to Applications.
"""

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
NAME = 'SANR'
BUNDLE_ID = 'local.sanr.app'
ICON = os.path.join(ROOT, 'icons', 'app.png')
# iconutil's names: icon_<size>x<size>[@2x].png
ICON_SIZES = (16, 32, 128, 256, 512)


def python_path():
    """The project's .venv Python if there is one, else this one."""
    venv = os.path.join(ROOT, '.venv', 'bin', 'python3')
    return venv if os.path.exists(venv) else sys.executable


def make_icns(png, icns):
    """An .icns of every size macOS asks for, made with iconutil."""
    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, NAME + '.iconset')
        os.mkdir(iconset)
        src = Image.open(png).convert('RGBA')
        for size in ICON_SIZES:
            for scale, suffix in ((1, ''), (2, '@2x')):
                px = size * scale
                src.resize((px, px), Image.LANCZOS).save(os.path.join(
                    iconset, 'icon_%dx%d%s.png' % (size, size, suffix)))
        subprocess.run(['iconutil', '-c', 'icns', iconset, '-o', icns],
                       check=True)


def build(dest):
    app = os.path.join(os.path.abspath(dest), NAME + '.app')
    if os.path.exists(app):
        # replace only an app this script made
        try:
            with open(os.path.join(app, 'Contents', 'Info.plist'), 'rb') as f:
                ours = plistlib.load(f).get('CFBundleIdentifier') == BUNDLE_ID
        except (OSError, plistlib.InvalidFileException):
            ours = False
        if not ours:
            sys.exit('%s exists and was not made by make_app.py: not '
                     'replaced' % app)
        shutil.rmtree(app)
    macos = os.path.join(app, 'Contents', 'MacOS')
    res = os.path.join(app, 'Contents', 'Resources')
    os.makedirs(macos)
    os.makedirs(res)

    # exec: the launched app process becomes Python, so macOS keeps one
    # Dock entry with this bundle's name and icon
    launcher = os.path.join(macos, NAME)
    with open(launcher, 'w') as f:
        f.write('#!/bin/sh\n'
                '# made by make_app.py: starts SANR from %s\n'
                'cd %s || exit 1\n'
                'exec %s %s "$@"\n'
                % (ROOT, shlex.quote(ROOT), shlex.quote(python_path()),
                   shlex.quote(os.path.join(ROOT, 'main.py'))))
    os.chmod(launcher, 0o755)

    make_icns(ICON, os.path.join(res, NAME + '.icns'))
    with open(os.path.join(app, 'Contents', 'Info.plist'), 'wb') as f:
        plistlib.dump({
            'CFBundleName': NAME,
            'CFBundleDisplayName': NAME,
            'CFBundleIdentifier': BUNDLE_ID,
            'CFBundleExecutable': NAME,
            'CFBundleIconFile': NAME + '.icns',
            'CFBundlePackageType': 'APPL',
            'CFBundleShortVersionString': '1.0',
            'NSHighResolutionCapable': True,
        }, f)
    return app


if __name__ == '__main__':
    print('Built', build(sys.argv[1] if len(sys.argv) > 1 else ROOT))
