"""Prints the build's version string: the project version, plus "+<short hash>" of the commit being built.

The hash is left out only when the commit carries a tag (CI_COMMIT_TAG, or an exact git describe match) equal to
the project version. meson's vcs_tag runs this at every build, so the string follows the checked-out commit without
a reconfigure.

Usage: vcs_version.py <project version> <source root>
"""
import os
import subprocess
import sys


def git(root, *args):
    result = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def main():
    if len(sys.argv) != 3:
        print("ERROR: vcs_version.py needs the project version and the source root", file=sys.stderr)
        return 1
    version, root = sys.argv[1], sys.argv[2]
    commit = git(root, "rev-parse", "--verify", "--short=8", "HEAD")
    if not commit:
        print("ERROR: could not read the current git hash", file=sys.stderr)
        return 1
    tag = os.environ.get("CI_COMMIT_TAG", "") or git(root, "describe", "--always", "--exact-match", commit)
    print(version if tag == version else version + "+" + commit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
