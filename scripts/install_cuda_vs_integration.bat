@echo off
echo Copying CUDA 12.6 VS integration files...

set SRC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\extras\visual_studio_integration\MSBuildExtensions
set DST=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\MSBuild\Microsoft\VC\v180\BuildCustomizations

copy "%SRC%\CUDA 12.6.props" "%DST%\"
copy "%SRC%\CUDA 12.6.targets" "%DST%\"
copy "%SRC%\CUDA 12.6.xml" "%DST%\"
copy "%SRC%\Nvda.Build.CudaTasks.v12.6.dll" "%DST%\"

echo Done. Verify:
dir "%DST%\CUDA*"
pause
