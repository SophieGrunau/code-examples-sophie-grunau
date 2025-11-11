# ============ Execute when closing Texifier ============

#!/bin/zsh
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# === Paths ===
SRC="/Users/leasophiegrunau/Desktop/PhD_Australia/Writing/Thesis/Write-ups"
DST="/Users/leasophiegrunau/Desktop/PhD_Australia/Writing/Thesis/Overleaf"
LIST_FILE="/Users/leasophiegrunau/Shortcuts/Latex_sync/overleaf_files.txt"
GIT="/usr/local/bin/git"

# Dialog helper (allows $var expansion)
say() { /usr/bin/osascript -e "display dialog \"$1\" buttons {\"OK\"} default button \"OK\""; }

# --- Basic checks
cd "$SRC" || { say "❌ Write-ups repo path not found: $SRC"; exit 1; }
[ -d ".git" ] || { say "❌ Not a git repo (Write-ups)."; exit 1; }
[ -d "$DST" ] || { say "❌ Overleaf folder not found: $DST"; exit 1; }
[ -f "$LIST_FILE" ] || { say "❌ List file missing: $LIST_FILE"; exit 1; }


# ---------- 1) Commit & push Write-ups (prompt only if changes) ----------
sleep 2  # let Texifier finish saving

# If no changes, stop here (do NOT sync Overleaf)
if ! $GIT status --porcelain | grep -q .; then
  say "✅ No changes to commit."
  exit 0
fi

# Ask for commit message
MSG=$(osascript <<'APPLESCRIPT'
  tell application "System Events"
    activate
    display dialog "Commit message for Write-ups:" default answer "" with title "Git Commit" buttons {"Cancel","OK"} default button "OK"
    set theText to text returned of the result
    return theText
  end tell
APPLESCRIPT
) || exit 0
[ -z "$MSG" ] && MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"

# Commit; if nothing to commit after all, exit (no sync)
$GIT add -A
$GIT commit -m "$MSG" || { say "Nothing to commit."; exit 0; }
$GIT push || { say "❌ Push failed - Write-ups (check SSH/token). Overleaf not synced."; exit 1; }


# ---------- 2) Mirror selected files into Overleaf ----------
# wipe DST except the keepers
find "$DST" -mindepth 1 -maxdepth 1 \
  ! -name '.git' \
  ! -name '.gitignore' \
  ! -name 'README.md' \
  ! -name 'Overleaf_sync' \
  -exec rm -rf {} +

# Copy each listed path into DST root (no new top-level dirs)
while IFS= read -r path; do
  [ -z "$path" ] && continue
  SRC_PATH="$SRC/$path"
  echo "Copying: $SRC_PATH -> $DST/"

  if [[ -d "$SRC_PATH" ]]; then
    rsync -av "$SRC_PATH/" "$DST/" # copy the CONTENTS of the dir into DST root
  elif [[ -f "$SRC_PATH" ]]; then  # copy the single file into DST root
    rsync -av "$SRC_PATH" "$DST/"
  else
    echo "⚠️  Not found: $SRC_PATH"
  fi
done < "$LIST_FILE"


# ---------- 3) Commit & push Overleaf ----------
cd "$DST" || { say "❌ Overleaf folder not found during commit step: $DST"; exit 1; }
if $GIT rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    $GIT add .
    $GIT commit -m "Auto-update Overleaf mirror on $(date '+%Y-%m-%d %H:%M:%S')" || true
    if $GIT push; then
        say "✅ Pushed Write-ups changes to GitHub & synced Overleaf successfully."
    else
        say "✅ Pushed Write-ups changes to GitHub, but ❌ Overleaf push failed (check SSH/token)."
        exit 1
      fi        
else
    say "✅ Pushed Write-ups changes, but Overleaf is not a git repo — skipping push."
    exit 0
fi
