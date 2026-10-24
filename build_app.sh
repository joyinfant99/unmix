#!/bin/bash
# Build dist/Unmix.app and dist/Unmix-<version>.pkg.
# Default: ad-hoc signed, not notarized (macOS shows the "Apple could not verify" warning on first open).
# For a pkg that opens with no warning, set all three (needs a paid Apple Developer account; see README "Signing"):
#   DEVELOPER_ID_APP="Developer ID Application: Your Name (TEAMID)"
#   DEVELOPER_ID_INSTALLER="Developer ID Installer: Your Name (TEAMID)"
#   NOTARY_PROFILE=unmix-notary     (created once with: xcrun notarytool store-credentials unmix-notary)
# APP_ONLY=1 ./build_app.sh skips the pkg: run dist/Unmix.app directly to test the bundle without installing.
# Models are NOT bundled; they download on first use into ~/Library/Application Support/Unmix.
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${1:-$(cat VERSION)}"
echo "$VERSION" > VERSION

SIGN=0
if [ -n "${DEVELOPER_ID_APP:-}" ] || [ -n "${DEVELOPER_ID_INSTALLER:-}" ] || [ -n "${NOTARY_PROFILE:-}" ]; then
  : "${DEVELOPER_ID_APP:?set DEVELOPER_ID_APP to sign for distribution}"
  : "${DEVELOPER_ID_INSTALLER:?set DEVELOPER_ID_INSTALLER to sign for distribution}"
  : "${NOTARY_PROFILE:?set NOTARY_PROFILE to notarize}"
  SIGN=1
fi
SIGN_ARGS=()
# PyInstaller signs every binary inside the bundle (with the hardened runtime) when given an identity
[ "$SIGN" = 1 ] && SIGN_ARGS=(--codesign-identity "$DEVELOPER_ID_APP" --osx-entitlements-file installer/entitlements.plist)

rm -rf build dist
.venv/bin/pyinstaller --noconfirm --windowed --name Unmix ${SIGN_ARGS[@]+"${SIGN_ARGS[@]}"} \
  --icon assets/Unmix.icns \
  --osx-bundle-identifier dev.unmix.app \
  --add-data index.html:. \
  --add-data VERSION:. \
  --add-data vendor/apollo:vendor/apollo \
  --add-binary vendor/rubberband/unmix-stretch:vendor/rubberband \
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

# Sign the finished bundle (the Info.plist edits above invalidated PyInstaller's signature)
if [ "$SIGN" = 1 ]; then
  codesign --force --deep --options runtime --timestamp --entitlements installer/entitlements.plist \
    --sign "$DEVELOPER_ID_APP" "$APP"
  codesign --verify --deep --strict --verbose=2 "$APP"
else
  # Ad-hoc sign (free; required for arm64 binaries to launch). Not the same as Apple notarization.
  codesign --force --deep --sign - "$APP"
fi

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
PB_SIGN=()
[ "$SIGN" = 1 ] && PB_SIGN=(--sign "$DEVELOPER_ID_INSTALLER" --timestamp)
productbuild --distribution dist/pkg/distribution.xml --resources installer/resources \
  --package-path dist/pkg ${PB_SIGN[@]+"${PB_SIGN[@]}"} "dist/Unmix-$VERSION.pkg"

if [ "$SIGN" = 1 ]; then
  echo "Notarizing (usually 2-10 minutes)..."
  xcrun notarytool submit "dist/Unmix-$VERSION.pkg" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "dist/Unmix-$VERSION.pkg"
  spctl --assess --type install --verbose "dist/Unmix-$VERSION.pkg"
fi
ls -lh "dist/Unmix-$VERSION.pkg"
