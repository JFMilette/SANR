"""
Build SANR-<version>-setup.exe, a Windows installer of SANR.  Run on
Windows (PyInstaller cannot build a Windows program elsewhere):

    pip install -r requirements.txt pyinstaller
    python make_windows_installer.py [--version 1.2] [--strict]

Also needs Inno Setup 6 (https://jrsoftware.org/isinfo.php; its ISCC.exe on
the PATH or in its default folder) and, for the compiled transfer-matrix
kernel, the Microsoft C++ Build Tools.  Without them the kernel is left
out and SANR uses its numpy product (same results, ~2-4x slower fits);
--strict stops instead (the GitHub workflow uses it).

Steps: the kernel (setup.py build_ext --inplace); icons/app.png as
SANR.ico; PyInstaller, one folder with SANR.exe, the icons beside the code
(style.py and main.py find them next to their own file); SANR.exe
--self-test (selftest.py); Inno Setup (installer/sanr.iss) packs the
folder.  Everything goes to build/windows/, the installer to
build/windows/installer/.

The installer needs no administrator rights (it installs for the user,
or for every user if run as administrator), adds a Start-menu entry, an
optional desktop icon and an uninstaller.  The version defaults to the
last git tag (v1.2 -> 1.2) or the date.
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'build', 'windows')
ICON_PNG = os.path.join(ROOT, 'icons', 'app.png')
# not used by the program: left out of it (pyqtgraph would take some
# along where they are installed; OpenGL is used, style.py)
EXCLUDE = ('matplotlib', 'tkinter', 'pytest', 'Cython', 'IPython',
           'PyQt5', 'PySide6', 'PySide2', 'torch', 'tensorboard',
           'tensorflow', 'pandas', 'jupyter_rfb', 'notebook')
ISCC = (r'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
        r'C:\Program Files\Inno Setup 6\ISCC.exe')


def run(cmd, **kw):
    print('>', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, **kw)


def version():
    try:
        tag = subprocess.run(['git', 'describe', '--tags', '--abbrev=0'],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        tag = ''
    return tag.lstrip('vV') or datetime.date.today().strftime('%Y.%m.%d')


def build_kernel(strict):
    r = run([sys.executable, 'setup.py', 'build_ext', '--inplace'])
    if r.returncode == 0:
        return
    msg = ('the compiled kernel could not be built (Microsoft C++ Build '
           'Tools?)')
    if strict:
        sys.exit(msg)
    print('WARNING: %s; SANR will use its numpy product' % msg)


def make_ico(path):
    from PIL import Image
    Image.open(ICON_PNG).convert('RGBA').save(
        path, sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])


def freeze(ico):
    cmd = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
           '--windowed', '--name', 'SANR', '--icon', ico,
           '--add-data', os.pathsep.join([os.path.join(ROOT, 'icons'),
                                          'icons']),
           '--distpath', os.path.join(OUT, 'dist'),
           '--workpath', os.path.join(OUT, 'work'),
           '--specpath', OUT]
    for m in EXCLUDE:
        cmd += ['--exclude-module', m]
    if run(cmd + [os.path.join(ROOT, 'main.py')]).returncode:
        sys.exit('PyInstaller failed')
    return os.path.join(OUT, 'dist', 'SANR')


def self_test(folder, strict):
    report = os.path.join(OUT, 'self-test.txt')
    if os.path.exists(report):
        os.remove(report)
    exe = os.path.join(folder, 'SANR.exe' if os.name == 'nt' else 'SANR')
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    code = run([exe, '--self-test', report], env=env, timeout=300).returncode
    text = open(report, encoding='utf-8').read() \
        if os.path.exists(report) else '(no report)'
    print(text)
    if code or not text.rstrip().endswith('OK'):
        if strict or 'compiled kernel' not in text:
            sys.exit('the built SANR failed its self-test')
        print('WARNING: self-test without the compiled kernel')


def installer(folder, ver, ico):
    iscc = shutil.which('iscc') or next(
        (p for p in ISCC if os.path.exists(p)), None)
    if iscc is None:
        sys.exit('Inno Setup 6 (ISCC.exe) not found: install it from '
                 'https://jrsoftware.org/isinfo.php')
    out = os.path.join(OUT, 'installer')
    r = run([iscc, '/DAppVersion=' + ver, '/DSourceDir=' + folder,
             '/DOutputDir=' + out, '/DIconFile=' + ico,
             os.path.join(ROOT, 'installer', 'sanr.iss')])
    if r.returncode:
        sys.exit('Inno Setup failed')
    return os.path.join(out, 'SANR-%s-setup.exe' % ver)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--version', help='default: the last git tag or the date')
    ap.add_argument('--strict', action='store_true',
                    help='stop if the compiled kernel cannot be built')
    ap.add_argument('--no-installer', action='store_true',
                    help='stop after the SANR folder and its self-test')
    a = ap.parse_args()
    if sys.platform != 'win32' and not a.no_installer:
        sys.exit('run this on Windows (or with --no-installer to try the '
                 'PyInstaller step here)')
    os.makedirs(OUT, exist_ok=True)
    build_kernel(a.strict)
    ico = os.path.join(OUT, 'SANR.ico')
    make_ico(ico)
    folder = freeze(ico)
    self_test(folder, a.strict)
    if a.no_installer:
        print('SANR built in', folder)
        return
    print('installer:', installer(folder, a.version or version(), ico))


if __name__ == '__main__':
    main()
