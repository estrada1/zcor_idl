#!/usr/bin/env bash
#
# Download CMS DE-SynPUF Sample 1 data for ZCoR-IPF reproduction.
#
# Source: CMS 2008-2010 Data Entrepreneurs' Synthetic Public Use File
# https://www.cms.gov/data-research/statistics-trends-and-reports/
#   medicare-claims-synthetic-public-use-files
#
# Downloads ~260 MB compressed, ~2.6 GB uncompressed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAW_DIR="$SCRIPT_DIR/data/raw"
EXTRACT_DIR="$RAW_DIR/extracted"

mkdir -p "$RAW_DIR" "$EXTRACT_DIR"

CMS_BASE="https://www.cms.gov/research-statistics-data-and-systems/downloadable-public-use-files/synpufs/downloads"
CMS_ALT="https://downloads.cms.gov/files"

# Files to download: (local_name  url)
declare -a FILES=(
  "beneficiary_2008.zip  ${CMS_BASE}/de1_0_2008_beneficiary_summary_file_sample_1.zip"
  "beneficiary_2009.zip  ${CMS_BASE}/de1_0_2009_beneficiary_summary_file_sample_1.zip"
  "inpatient.zip         ${CMS_BASE}/de1_0_2008_to_2010_inpatient_claims_sample_1.zip"
  "outpatient.zip        ${CMS_BASE}/de1_0_2008_to_2010_outpatient_claims_sample_1.zip"
  "carrier_1a.zip        ${CMS_ALT}/DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.zip"
  "carrier_1b.zip        ${CMS_ALT}/DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.zip"
)

# Documentation
declare -a DOCS=(
  "codebook.pdf  https://www.cms.gov/files/document/de-10-codebook.pdf-0"
  "dug.pdf       ${CMS_BASE}/synpuf_dug.pdf"
)

echo "============================================"
echo "CMS DE-SynPUF Data Download"
echo "============================================"
echo ""
echo "Destination: $RAW_DIR"
echo ""

download_file() {
  local name="$1"
  local url="$2"
  local dest="$RAW_DIR/$name"

  if [ -f "$dest" ]; then
    echo "  [skip] $name (already exists)"
    return 0
  fi

  echo "  [downloading] $name ..."
  if curl -fSL --progress-bar -o "$dest.tmp" "$url"; then
    mv "$dest.tmp" "$dest"
    echo "  [done] $name ($(du -h "$dest" | cut -f1))"
  else
    rm -f "$dest.tmp"
    echo "  [FAILED] $name — check URL or network"
    return 1
  fi
}

echo "Downloading claims data..."
for entry in "${FILES[@]}"; do
  name=$(echo "$entry" | awk '{print $1}')
  url=$(echo "$entry" | awk '{print $2}')
  download_file "$name" "$url"
done

echo ""
echo "Downloading documentation..."
for entry in "${DOCS[@]}"; do
  name=$(echo "$entry" | awk '{print $1}')
  url=$(echo "$entry" | awk '{print $2}')
  download_file "$name" "$url"
done

echo ""
echo "Extracting zip files..."
for zipfile in "$RAW_DIR"/*.zip; do
  [ -f "$zipfile" ] || continue
  base=$(basename "$zipfile")
  echo "  [extracting] $base ..."
  unzip -o -q "$zipfile" -d "$EXTRACT_DIR"
done

echo ""
echo "============================================"
echo "Download complete!"
echo "============================================"
echo ""
echo "Extracted files:"
ls -lh "$EXTRACT_DIR"/*.csv 2>/dev/null || echo "  (none found)"
echo ""
echo "Next step: run the preprocessing pipeline:"
echo "  uv run python preprocess_cms.py"
