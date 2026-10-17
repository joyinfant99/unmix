#!/bin/bash
# Build dist/Unmix.app and dist/Unmix-<version>.pkg (not Apple-notarized; ad-hoc signed so it runs on Apple Silicon).
# APP_ONLY=1 ./build_app.sh skips the pkg: run dist/Unmix.app directly to test the bundle without installing.
# Models are NOT bundled; they download on first use into ~/Library/Application Support/Unmix.
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${1:-$(cat VERSION)}"
echo "$VERSION" > VERSION

rm -rf build dist
.venv/bin/pyinstaller --noconfirm --windowed --name Unmix \
  --icon assets/Unmix.icns \
  --osx-bundle-identifier dev.unmix.app \
  --add-data index.html:. \
  --add-data VERSION:. \
  --add-data vendor/apollo:vendor/apollo \
  --collect-all audio_separator \
  --collect-all webview \
  --collect-all yt_dlp \
  --collect-all imageio_ffmpeg \
  --collect-submodules uvicorn \
  --hidden-import app \
  --exclude-module tkinter \
  desktop.py

APP=dist/Unmix.app
for KEY in CFBundleShortVersionString CFBundleVersion; do  # PyInstaller writes 0.0.0 first; Set when present, Add otherwise
  /usr/libexec/PlistBuddy -c "Set :$KEY $VERSION" "$APP/Contents/Info.plist" 2>/dev/null \
    || /usr/libexec/PlistBuddy -c "Add :$KEY string $VERSION" "$APP/Contents/Info.plist"
done
/usr/libexec/PlistBuddy -c "Add :NSDesktopFolderUsageDescription string Unmix reads audio files you choose." "$APP/Contents/Info.plist" 2>/dev/null || true

# Register the .umx project type: a package (looks like one file in Finder) that opens in Unmix
.venv/bin/python - "$APP/Contents/Info.plist" <<'PY'
import plistlib, sys
path = sys.argv[1]
info = plistlib.load(open(path, "rb"))
info["CFBundleDocumentTypes"] = [{"CFBundleTypeName": "Unmix Project", "CFBundleTypeRole": "Editor", "LSHandlerRank": "Owner",
                                  "LSTypeIsPackage": True, "LSItemContentTypes": ["dev.unmix.project"], "CFBundleTypeIconFile": "Unmix.icns"}]
info["UTExportedTypeDeclarations"] = [{"UTTypeIdentifier": "dev.unmix.project", "UTTypeDescription": "Unmix Project",
                                       "UTTypeConformsTo": ["com.apple.package"], "UTTypeIconFile": "Unmix.icns",
                                       "UTTypeTagSpecification": {"public.filename-extension": ["umx"]}}]
plistlib.dump(info, open(path, "wb"))
PY

# Ad-hoc sign (free; required for arm64 binaries to launch). Not the same as Apple notarization.
codesign --force --deep --sign - "$APP"

if [ "${APP_ONLY:-}" = "1" ]; then echo "Built $APP (no pkg). Run: open $APP"; exit 0; fi

# Installer package: runs the system check (installer/distribution.xml.in) before anything is installed
rm -rf dist/pkg && mkdir -p dist/pkg
# pkgbuild rejects the bundle ("not a valid bundle component") when it sits under ~/Desktop or ~/Documents,
# so package a copy from a temporary folder
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/Unmix.app"
pkgbuild --component "$STAGE/Unmix.app" --install-location /Applications --ownership recommended \
  --scripts installer/scripts --identifier dev.unmix.app --version "$VERSION" dist/pkg/Unmix-component.pkg
sed "s/@VERSION@/$VERSION/" installer/distribution.xml.in > dist/pkg/distribution.xml
productbuild --distribution dist/pkg/distribution.xml --resources installer/resources \
  --package-path dist/pkg "dist/Unmix-$VERSION.pkg"
ls -lh "dist/Unmix-$VERSION.pkg"
