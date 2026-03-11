#!/bin/bash
# Clear Bluetooth cache on macOS - fixes pairing/connection issues (e.g. GIK devices)
# WARNING: Removes ALL paired Bluetooth devices - you'll need to re-pair everything.

set -e
echo "=== Bluetooth Cache Reset for macOS ==="
echo ""
echo "This will:"
echo "  1. Remove Bluetooth preference files"
echo "  2. Restart the Bluetooth daemon"
echo "  3. Require re-pairing of all Bluetooth devices"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo "Removing Bluetooth preferences..."
sudo rm -f /Library/Preferences/com.apple.Bluetooth.plist 2>/dev/null || true
rm -f ~/Library/Preferences/com.apple.Bluetooth.plist 2>/dev/null || true
rm -f ~/Library/Preferences/ByHost/com.apple.Bluetooth.*.plist 2>/dev/null || true

echo "Restarting Bluetooth daemon..."
sudo pkill bluetoothd 2>/dev/null || true
# bluetoothd auto-restarts

echo "Clearing preference cache..."
sudo killall cfprefsd 2>/dev/null || true

echo ""
echo "Done. Bluetooth cache cleared."
echo "Turn Bluetooth off and on in System Settings, then re-pair your GIK devices."
echo "A full restart may help if issues persist."
