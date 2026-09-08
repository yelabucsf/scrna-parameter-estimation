"""Shared bundle machinery for the per-figure data manifests.

Each `figureN/data_manifest.py` declares *what* its figure needs; this module does the
mechanical part — building the panel-organized tree, and packing it into the archive that
gets published so a reader can reproduce one figure without the 346 GB source volume.

The archive's single top-level directory is `figureN_data/`, which is what makes the
download a one-knob affair: `config.FIGUREN_DATA` already defaults to
`MEMENTO_DATA_PATH + 'figureN_data/'`, so extracting into any directory and pointing
MEMENTO_DATA_PATH at it is enough. Renaming that directory breaks every figure README.
"""

import hashlib
import os
import shutil
import sys
import tarfile

REQUIRED, PROVENANCE = 'required', 'provenance'
CHUNK = 1 << 20


def build(rows, root, copy, require_all=False):
    """Materialize `rows` under `root`, as copies or as symlinks.

    Returns (built, skipped). Sources that do not exist are skipped rather than raising,
    because a partially-synced volume is a normal state to want to inspect — but
    `require_all` turns a missing REQUIRED-tier source into an error, which is what
    archiving needs: a bundle silently missing a panel's input is worse than no bundle.
    """
    present = [r for r in rows if os.path.exists(r[3])]
    skipped = [r for r in rows if not os.path.exists(r[3])]

    if require_all:
        fatal = [r for r in skipped if r[1] == REQUIRED]
        if fatal:
            listing = '\n'.join(f'  {r[3]}' for r in fatal[:10])
            more = f'\n  ... and {len(fatal) - 10} more' if len(fatal) > 10 else ''
            raise SystemExit(
                f'refusing to build: {len(fatal)} required file(s) missing from the '
                f'source volume:\n{listing}{more}')

    for _, _, dest, src in present:
        target = os.path.join(root, dest)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.lexists(target):
            os.remove(target)
        if copy:
            shutil.copy2(src, target)
        else:
            os.symlink(os.path.realpath(src), target)

    print(f'{"copied" if copy else "linked"} {len(present)} files into {root}')
    if skipped:
        print(f'skipped {len(skipped)} missing source(s):')
        for row in skipped[:10]:
            print(f'  {row[1]:<10} {row[3]}')
        if len(skipped) > 10:
            print(f'  ... and {len(skipped) - 10} more')
    return present, skipped


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def write_archive(root, out_path):
    """Tar-gzip `root` as a single top-level directory, and write a .sha256 beside it.

    The checksum covers the archive rather than its members: it is there so a reader can
    tell a truncated download from a good one with `sha256sum -c`, which needs no code
    from this repository.
    """
    root = os.path.abspath(root.rstrip('/'))
    name = os.path.basename(root)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)

    print(f'packing {root} -> {out_path}')
    with tarfile.open(out_path, 'w:gz') as tar:
        tar.add(root, arcname=name)

    digest = sha256(out_path)
    checksum_path = out_path + '.sha256'
    with open(checksum_path, 'w') as handle:
        handle.write(f'{digest}  {os.path.basename(out_path)}\n')

    size = os.path.getsize(out_path)
    print(f'wrote {out_path} ({size / 1e9:.2f} GB)')
    print(f'wrote {checksum_path}')
    print(f'  sha256 {digest}')
    return out_path, digest


def add_arguments(parser, commands=('check', 'link', 'bundle', 'archive')):
    parser.add_argument('command', choices=list(commands))
    parser.add_argument('--root', default=None,
                        help='organized tree to build, or to check instead of the source '
                             'volume. Required for bundle and archive.')
    parser.add_argument('--out', default=None,
                        help='archive path to write (archive only). Defaults to '
                             '<root>.tar.gz')
    return parser


def dispatch(args, entries, check, link_root):
    """Run the subcommand. `link_root` is where a bare `link` builds its symlink tree."""
    if args.command == 'check':
        raise SystemExit(0 if check(args.root) else 1)

    if args.command == 'link':
        build(entries(), args.root or link_root, copy=False)
        return

    # bundle and archive write real copies, so they never default to link_root: doing so
    # would overwrite the symlink tree on the source volume with GBs of duplicates.
    if not args.root:
        raise SystemExit(f'{args.command} requires --root DIR (a staging directory). '
                         'It writes real copies, so it will not default to the source '
                         'volume.')

    build(entries(), args.root, copy=True, require_all=True)
    if args.command == 'archive':
        write_archive(args.root, args.out or args.root.rstrip('/') + '.tar.gz')


def add_publication_to_path():
    """Let a figure's data_manifest.py import this module."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
