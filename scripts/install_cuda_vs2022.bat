@echo off
echo Copying CUDA 12.6 integration to VS 2022 Build Tools...

set SRC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\extras\visual_studio_integration\MSBuildExtensions
set DST=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations

xcopy "%SRC%\*.*" "%DST%\" /Y

echo.
echo Verify:
dir "%DST%\CUDA*"
pause
