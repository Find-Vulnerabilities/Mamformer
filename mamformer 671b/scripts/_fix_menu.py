"""One-time fix: repair run.bat menu alignment."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
bat_path = root / "run.bat"

with open(bat_path, "rb") as f:
    data = f.read()

# Fix 1: replace literal CR (0x0D) followed by 'aw' with 'raw\'
data = data.replace(b"data\raw", b"data\\raw\\")

# Fix 2: ensure the line is complete
# The broken line should read: data\raw\)
# If it still looks wrong, manually inject the correct line
old = b"echo  ^|   [1] Auto clean + classify (scan data\\raw\\"
new = b"echo  ^|   [1] Auto clean + classify (scan data\\raw\\)             ^|"
if old in data:
    data = data.replace(old, new, 1)

with open(bat_path, "wb") as f:
    f.write(data)

print("Menu fixed.")
