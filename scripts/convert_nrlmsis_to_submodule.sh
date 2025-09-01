#!/usr/bin/env bash
set -euo pipefail

# Convert the local nrlmsis2.1_cpp folder into a git submodule pointing to the
# Stuttgart repo (branch master). This script performs the necessary git steps
# in your working copy. It will NOT push anything – you commit/push afterwards.

URL_DEFAULT="https://git.ins.uni-stuttgart.de/roman.krueckel/nrlmsis2.1_cpp"
TARGET_DIR="nrlmsis2.1_cpp"
BRANCH="master"

url="${1:-$URL_DEFAULT}"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "[ERROR] This is not a git repository. Run in your repo root." >&2
  exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
  echo "[INFO] $TARGET_DIR does not exist. Adding submodule fresh..."
  git submodule add -b "$BRANCH" "$url" "$TARGET_DIR"
  git submodule update --init --recursive
  git add .gitmodules "$TARGET_DIR"
  echo "[OK] Submodule added. Commit with: git commit -m 'Add nrlmsis2.1_cpp as submodule'"
  exit 0
fi

# If it's already a git repo, assume it's a submodule or separate clone
if [ -d "$TARGET_DIR/.git" ] || git -C "$TARGET_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  echo "[INFO] $TARGET_DIR looks like a git repo already. If it's not linked as a submodule, remove it first."
  exit 0
fi

echo "[INFO] Detected vendored $TARGET_DIR (not a git repo). Converting to submodule..."
read -p "Backup the current folder to ${TARGET_DIR}_backup before removal? [Y/n] " ans
ans=${ans:-Y}
if [[ "$ans" =~ ^[Yy]$ ]]; then
  bk="${TARGET_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
  echo "[INFO] Backing up to $bk"
  cp -a "$TARGET_DIR" "$bk"
fi

git rm -r "$TARGET_DIR"
git commit -m "Remove vendored $TARGET_DIR in preparation for submodule"
git submodule add -b "$BRANCH" "$url" "$TARGET_DIR"
git submodule update --init --recursive
git add .gitmodules "$TARGET_DIR"
echo "[OK] Submodule added. Commit with: git commit -m 'Add nrlmsis2.1_cpp as submodule'"

