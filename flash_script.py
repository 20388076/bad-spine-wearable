import os
from SCons.Script import Import

Import("env")
# Get the user choice from platformio.ini (string or number)
choice = env.GetProjectOption("custom_choice")
proj_dir = env['PROJECT_DIR']
src_dir = os.path.join(proj_dir, "src")

# List all .cpp files in src/
files = sorted(f for f in os.listdir(src_dir) if f.endswith(".cpp"))
if not files:
    print("No .cpp files found in src/")
    Return()

# Print available files for reference
print("Available .cpp files:")
for idx, fname in enumerate(files):
    print("  [{}] {}".format(idx, fname))

# Determine selected file
target = None
if choice is None:
    # If no choice given, error out or pick a default
    print("No custom_choice specified; aborting")
    Exit(1)
else:
    choice = str(choice).strip()
    if choice.isdigit():
        idx = int(choice)
        if 0 <= idx < len(files):
            target = files[idx]
    else:
        # Allow specifying filename (with or without .cpp)
        target = choice if choice.endswith(".cpp") else choice + ".cpp"
        if target not in files:
            target = None

# Check result
if target is None:
    print("Invalid custom_choice '{}'; must be an index or filename".format(choice))
    Exit(1)

print("Building only: {}".format(target))
# Set SRC_FILTER to exclude all but the chosen file
env.Replace(SRC_FILTER=["-<*>", "+<{}>".format(target)])