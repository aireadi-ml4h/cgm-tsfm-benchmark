#!/bin/bash
#SBATCH --job-name=cgmacros_download
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/%x_%j.out

# Re-downloads CGMacros (PhysioNet DOI 10.13026/3z8q-x658, open access, no
# credentialing) into THIS user's own scratch. The existing copy the meal
# detector was built against lives in /path/to/work (mode 0700,
# unreadable here) and is subject to the 90-day scratch purge, so the
# reproduction arm cannot depend on it. CPU-only; not run on the login node.
set -euo pipefail
DEST="$AIREADI_SCRATCH/cgmacros/1.0.0"
mkdir -p "$DEST"

ml load system aws-cli
aws s3 sync --no-sign-request s3://physionet-open/cgmacros/1.0.0/ "$DEST/"

cd "$DEST"
sha256sum -c SHA256SUMS.txt || echo "WARN: checksum output above (some entries may not apply to synced set)"

mkdir -p "$DEST/extracted"
unzip -q -o CGMacros_dateshifted365.zip -d "$DEST/extracted"
echo "extracted top-level:"
ls "$DEST/extracted" | head
echo "CGMACROS_DOWNLOAD_OK"
