"""
Build SANR-<version>-macos-<arm64|x86_64>.dmg, a disk image of SANR.app (a
standalone macOS application, Python and every package inside) for the
architecture of the Python running it.  Run on macOS:

    pip install -r requirements.txt pyinstaller
    python make_mac_installer.py [--version 1.2] [--strict]

The compiled transfer-matrix kernel needs the Xcode command-line tools
(xcode-select --install); without them it is left out and SANR uses its
numpy product, unless --strict.

Steps, as make_windows_installer.py: the kernel; PyInstaller, SANR.app
(icons/app.png as its icon); SANR.app's --self-test (selftest.py); the
.dmg (hdiutil) holding SANR.app and a link to Applications to drag it to.
Everything goes to build/macos/, the image to build/macos/installer/.

The app is not signed with an Apple developer identity (PyInstaller signs
it ad hoc): the first time, macOS refuses to open it until it is allowed
in System Settings -> Privacy & Security -> Open Anyway (or right-click ->
Open on older macOS).  This is not the developer app of make_app.py,
which runs this folder's own .venv.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile

import make_windows_installer as common

ROOT = common.ROOT
OUT = os.path.join(ROOT, 'build', 'macos')


def make_dmg(app, ver):
    arch = platform.machine().lower()
    out = os.path.join(OUT, 'installer')
    os.makedirs(out, exist_ok=True)
    dmg = os.path.join(out, 'SANR-%s-macos-%s.dmg' % (ver, arch))
    if os.path.exists(dmg):
        os.remove(dmg)
    with tempfile.TemporaryDirectory() as stage:
        shutil.copytree(app, os.path.join(stage, 'SANR.app'), symlinks=True)
        os.symlink('/Applications', os.path.join(stage, 'Applications'))
        r = common.run(['hdiutil', 'create', '-volname', 'SANR %s' % ver,
                        '-srcfolder', stage, '-ov', '-format', 'UDZO', dmg])
    if r.returncode:
        sys.exit('hdiutil failed')
    return dmg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--version', help='default: the last git tag or the date')
    ap.add_argument('--strict', action='store_true',
                    help='stop if the compiled kernel cannot be built')
    ap.add_argument('--no-dmg', action='store_true',
                    help='stop after SANR.app and its self-test')
    a = ap.parse_args()
    if sys.platform != 'darwin':
        sys.exit('run this on macOS')
    os.makedirs(OUT, exist_ok=True)
    common.build_kernel(a.strict)
    dist = common.freeze(common.ICON_PNG, OUT,
                         ['--osx-bundle-identifier', 'ca.uottawa.sanr'])
    app = os.path.join(dist, 'SANR.app')
    common.self_test(os.path.join(app, 'Contents', 'MacOS', 'SANR'),
                     a.strict, OUT)
    if a.no_dmg:
        print('SANR built in', app)
        return
    print('disk image:', make_dmg(app, a.version or common.version()))


if __name__ == '__main__':
    main()
