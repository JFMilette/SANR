"""
Session files as a .zip: session.json, and under data/ a copy of
every file the session reads again on opening (the imported channels' data
files, and their Q files when they have one), byte for byte.  Such a session
opens anywhere, without the folders the data came from.

A file is referenced in the session dict as tabs.experimental.file_ref gives
it: {'path': absolute, 'relative': to the session folder}.  write_session
copies each such file once into data/ (two files of the same name become
name.dat, name_2.dat) and adds 'embedded': its name in the zip.
read_session writes the embedded files to a folder of its own (kept until
the program ends) and points each reference's 'path' there, so the copy in
the session is what is read, never a file that changed since.

The ended fit and DREAM tabs (the session's 'fit_tabs') get a file each
under tabs/, numbered in the tabs' order and named after the algorithm
(tab_label), so session.json stays light: a fit tab's whole state as
tabs/<n>_<algorithm>.json (e.g. tabs/1_L-BFGS-B.json), a DREAM run as the
file "Save chains…" writes, tabs/<n>_DREAM.npz (its header and status stay
in session.json).  Their entries in session.json say which file: 'file' and
'npz_file'.  read_session puts them back as a .json session has them.

A plain .json session (files by path, as before) is still read and written.
"""

import atexit
import base64
import json
import os
import re
import shutil
import tempfile
import zipfile


def sessions_dir():
    """Where the session dialogs (and Save chains) open first: sessions/
    of the repo, or Documents/SANR for a built program (its folder is not
    writable), the home folder if that cannot be made."""
    import sys
    if getattr(sys, 'frozen', False):
        d = os.path.join(os.path.expanduser('~'), 'Documents', 'SANR')
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            d = os.path.expanduser('~')
        return d
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'sessions')

SESSION_NAME = 'session.json'
DATA_DIR = 'data/'
TABS_DIR = 'tabs/'
_EXTRACTED = []                         # folders read_session made


def file_refs(d):
    """Every file reference (a dict with 'path' and 'relative') in d."""
    if isinstance(d, dict):
        if isinstance(d.get('path'), str) and isinstance(d.get('relative'),
                                                         str):
            yield d
            return
        for v in d.values():
            yield from file_refs(v)
    elif isinstance(d, list):
        for v in d:
            yield from file_refs(v)


def is_zip_session(path):
    return zipfile.is_zipfile(path)


def tab_label(t):
    """The algorithm of a saved fit or DREAM tab, for its file name: the
    first word of its tab name ('L-BFGS-B 2 · film' -> 'L-BFGS-B'), else
    its method or kind; only letters, digits, '.', '-' and '_'."""
    name = (t.get('tab_name') or '').split(' ')[0] or t.get('method')         or t.get('kind') or 'tab'
    return re.sub(r'[^A-Za-z0-9._-]', '_', str(name)) or 'tab'


def only_in_session(d):
    """The referenced files of d that are copies read_session made (gone
    when the program ends): a .json session would lose them."""
    return [r['relative'] for r in file_refs(d)
            if any(os.path.abspath(r['path']).startswith(f + os.sep)
                   for f in _EXTRACTED)]


def write_session(path, d, default=None):
    """Write session dict d (changed: references get 'embedded') to `path`
    as a zip with the referenced files.  Written beside and then renamed,
    so a failure leaves any earlier file as it was."""
    tabs, files = [], {}                # session.json entries, zip files
    for k, t in enumerate(d.get('fit_tabs') or [], 1):
        if t.get('kind') == 'dream' and 'npz' in t:
            name = '%s%d_%s.npz' % (TABS_DIR, k, tab_label(t))
            files[name] = base64.b64decode(t['npz'])
            t = {key: v for key, v in t.items() if key != 'npz'}
            t['npz_file'] = name
        elif t.get('kind') == 'fit':
            name = '%s%d_%s.json' % (TABS_DIR, k, tab_label(t))
            files[name] = json.dumps(t, indent=1,
                                     default=default).encode('utf-8')
            t = dict(kind='fit', file=name, tab_name=t.get('tab_name'))
        tabs.append(t)
    if 'fit_tabs' in d:
        d['fit_tabs'] = tabs
    names = {}                          # absolute path -> name in the zip
    used = set()
    for ref in file_refs(d):
        src = os.path.abspath(ref['path'])
        if src not in names:
            if not os.path.isfile(src):
                raise OSError('not found: %s' % src)
            stem, ext = os.path.splitext(os.path.basename(src))
            name, n = stem + ext, 1
            while name.lower() in used:
                n += 1
                name = '%s_%d%s' % (stem, n, ext)
            used.add(name.lower())
            names[src] = name
        ref['embedded'] = DATA_DIR + names[src]
    tmp = path + '.part'
    try:
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr(SESSION_NAME, json.dumps(d, indent=2, default=default))
            for src, name in names.items():
                z.write(src, DATA_DIR + name)
            for name, data in files.items():
                z.writestr(name, data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_session(path):
    """Session dict of a .zip session, its references pointing to the
    embedded copies (written to a folder of their own).  Returns
    (d, folder)."""
    with zipfile.ZipFile(path) as z:
        if SESSION_NAME not in z.namelist():
            raise ValueError('not a SANR session: no %s at the top of the '
                             'zip' % SESSION_NAME)
        d = json.loads(z.read(SESSION_NAME).decode('utf-8'))
        folder = tempfile.mkdtemp(prefix='sanr_session_')
        _EXTRACTED.append(folder)
        tabs = []
        for t in d.get('fit_tabs') or []:
            if isinstance(t.get('file'), str):
                t = json.loads(z.read(_member(t['file'], TABS_DIR))
                               .decode('utf-8'))
            elif isinstance(t.get('npz_file'), str):
                data = z.read(_member(t['npz_file'], TABS_DIR))
                t = {key: v for key, v in t.items() if key != 'npz_file'}
                t['npz'] = base64.b64encode(data).decode('ascii')
            tabs.append(t)
        if 'fit_tabs' in d:
            d['fit_tabs'] = tabs
        done = {}
        for ref in file_refs(d):
            name = ref.get('embedded')
            if not isinstance(name, str):
                continue
            base = _member(name, DATA_DIR)[len(DATA_DIR):]
            if name not in done:
                out = os.path.join(folder, base)
                with open(out, 'wb') as f:
                    f.write(z.read(name))
                done[name] = out
            ref['path'] = done[name]
            ref['relative'] = base
    return d, folder


def _member(name, folder):
    """name, a file of the zip: only a plain file name under `folder`,
    nothing that would lead outside it."""
    base = name[len(folder):]
    if not name.startswith(folder) or not base or \
            base != os.path.basename(base) or base in ('.', '..') \
            or '\\' in base:
        raise ValueError('bad file name in the session: %r' % name)
    return name


@atexit.register
def _clean():
    for folder in _EXTRACTED:
        shutil.rmtree(folder, ignore_errors=True)
