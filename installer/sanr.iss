; Inno Setup 6 script of the SANR installer: packs the PyInstaller folder
; that make_windows_installer.py builds (it passes the four defines below).

#ifndef AppVersion
  #define AppVersion "0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\build\windows\dist\SANR"
#endif
#ifndef OutputDir
  #define OutputDir "..\build\windows\installer"
#endif
#ifndef IconFile
  #define IconFile "..\build\windows\SANR.ico"
#endif

[Setup]
; the same AppId for every version: a new one installs over the old one
AppId={{6F3C2A51-8E47-4B9D-9C1E-5A2D7B0E4F18}
AppName=SANR
AppVersion={#AppVersion}
AppVerName=SANR {#AppVersion}
AppPublisher=JF Milette
DefaultDirName={autopf}\SANR
DefaultGroupName=SANR
DisableProgramGroupPage=yes
; installs for the user without administrator rights, or for everyone
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=SANR-{#AppVersion}-setup
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\SANR.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; files of an older version that this one no longer has
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{autoprograms}\SANR"; Filename: "{app}\SANR.exe"
Name: "{autodesktop}\SANR"; Filename: "{app}\SANR.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\SANR.exe"; Description: "{cm:LaunchProgram,SANR}"; Flags: nowait postinstall skipifsilent
