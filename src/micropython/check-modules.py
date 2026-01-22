import sys
import os

# 1. Check Version info
print(f"Platform:       {sys.platform}")
print(f"Version:        {sys.version}")
print(f"Implementation: {sys.implementation.name} {sys.implementation.version}")

# List all available modules
print("\n--- All Available Modules ---")
# This command prints directly to the console
help('modules')

print("=" * 30)