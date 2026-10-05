#!/usr/bin/env bash
# Run once, after filling the author fields in CITATION.cff, LICENSE-CODE and README.md.
# 1. writes CHECKSUMS.sha256 for every file of the package;
# 2. commits the package to a local git repository (the "content commit");
# 3. writes RELEASE.txt with that commit hash and the SHA-256 of CHECKSUMS.sha256,
#    and commits it on top. RELEASE.txt therefore identifies the content commit,
#    i.e. the parent of the final HEAD. The Zenodo DOI identifies the archived version.
set -euo pipefail
cd "$(dirname "$0")"
if grep -q TODO_AUTHOR_NAME CITATION.cff LICENSE-CODE; then echo "fill the author fields first"; exit 1; fi
if command -v sha256sum >/dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
find . -type f ! -path './.git/*' ! -name CHECKSUMS.sha256 ! -name RELEASE.txt -print0 | LC_ALL=C sort -z | xargs -0 $SHA > CHECKSUMS.sha256
git init -q 2>/dev/null || true
git add -A && git commit -q -m "Replication package, submitted version"
H=$(git rev-parse HEAD)
C=$($SHA CHECKSUMS.sha256 | cut -d' ' -f1)
cat > RELEASE.txt <<EOF
Content commit:              $H
SHA-256 of CHECKSUMS.sha256: $C
Recorded (UTC):              $(date -u +%Y-%m-%dT%H:%M:%SZ)
The content commit contains every file listed in CHECKSUMS.sha256; this file was
committed on top of it. The Zenodo DOI of the upload identifies the archived version.
Verify an extracted copy with:  sha256sum -c CHECKSUMS.sha256   (macOS: shasum -a 256 -c CHECKSUMS.sha256)
EOF
git add RELEASE.txt && git commit -q -m "Record release identifiers"
cat RELEASE.txt
echo "now zip this directory (excluding .git if you prefer), upload it to Zenodo, and put the DOI in the manuscript"
