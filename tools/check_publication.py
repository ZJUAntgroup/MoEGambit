"""Check tracked source and image metadata for accidental private deployment data.

This is a lightweight release guard, not a general-purpose secret scanner.
Public upstream copyright notices and example loopback addresses are retained.
"""

from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "personal home directory": rb"/(?:Users|home)/[A-Za-z0-9_.-]+/",
    "internal cluster mount": rb"/mnt/ais-[A-Za-z0-9_.-]+/",
    "private code host": rb"code[.]alipay[.]com",
    "private key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "GitHub token": rb"(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})",
    "AWS access key": rb"AKIA[0-9A-Z]{16}",
}


def main():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    failures = []
    for name in filter(None, names):
        # Unmodified upstream examples contain public author paths; scope personal
        # home checks to the project, but scan every tracked file for credentials.
        vendored = name.startswith(("DeepSpeed/", "Megatron-LM/"))
        path = ROOT / name
        if path.is_symlink():
            data = str(path.readlink()).encode()
        elif path.is_file():
            data = path.read_bytes()
        else:
            continue
        for label, pattern in PATTERNS.items():
            if vendored and label == "personal home directory":
                continue
            if re.search(pattern, data):
                failures.append(f"{name}: {label}")
    if failures:
        print("\n".join(failures))
        return 1
    print("Publication scan passed for tracked source and resources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
