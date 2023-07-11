# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['backend.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\user\\anaconda3\\envs\\GIS_10_windows_package\\Lib\\site-packages\\shapely', '.\\shapely'),('C:\\Users\\user\\Desktop\\gis_project\\Forms\\App\\red_lines_ico_newer.ico','.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Red lines',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['red_lines_ico_newer.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Red lines',
)
