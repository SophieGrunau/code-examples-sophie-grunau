```markdown
# Workflow Automation: LaTeX Git Integration

**Author:** Lea Sophie Grunau

Automated Git commit and Overleaf sync triggered when closing Texifier (LaTeX editor) 
on macOS. Built to prevent forgetting to commit LaTeX source files after writing sessions.

## How It Works

The script is integrated into macOS Shortcuts to run automatically when Texifier closes:

1. Checks for uncommitted changes in the LaTeX repository
2. If changes exist, prompts for a commit message via a dialog box
3. Commits and pushes to GitHub
4. Mirrors selected files into a local Overleaf sync folder

## Contents

- `ClosingTexifier.sh` — main automation script
- `overleaf_files.txt` — list of files/folders to mirror to Overleaf

## Setup

**Note:** This script is macOS-specific and requires:
- Texifier (LaTeX editor)
- Git configured with SSH or token
- Rsync
- A local Overleaf sync folder

The script is not designed to run portably — paths and the macOS Shortcuts 
integration are specific to the author's system. It is included here as a 
code example demonstrating shell scripting and workflow automation.

## Dependencies

- macOS Zsh shell
- Git
- Rsync
- AppleScript (`osascript`) for dialogs
- `overleaf_files.txt` listing files/folders to mirror
```