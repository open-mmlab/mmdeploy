Import-Module .\utils.psm1

#read configuration file
$config_path = '../conf/win_default.config'
$conf = ReadConfig $config_path
$cuda_version=$conf.cuda_version
Write-Host "cuda_version=$cuda_version"
$codebase_list=$conf.codebase_list
Write-Host "codebase_list=$codebase_list"
$exec_performance=$conf.exec_performance
Write-Host "exec_performance=$exec_performance"
$max_job_nums=$conf.max_job_nums
Write-Host "max_job_nums=$max_job_nums"
$mmdeploy_branch=$conf.mmdeploy_branch
Write-Host "mmdeploy_branch=$mmdeploy_branch"
$repo_url=$conf.repo_url
Write-Host "repo_url=$repo_url"

SwitchCudaVersion $cuda_version

if ( $exec_performance -eq "y" ) {
    $exec_performance='--performance'
}else {
    $exec_performance=$null
}

$codebase_fullname_opt = @{
    "mmdet" = "mmdetection";
    "mmcls" = "mmclassification";
    "mmdet3d" = "mmdetection3d";
    "mmedit" = "mmediting";
    "mmocr" = "mmocr";
    "mmpose" = "mmpose";
    "mmrotate" = "mmrotate";
    "mmseg" = "mmsegmentation"
}

$env:WORKSPACE="C:\Users\HZJ\Desktop\mmdeploy_windows"
$env:OPENCV_DIR=(Join-PATH $env:WORKSPACE opencv\4.6.0\build\x64\vc15)
$env:TENSORRT_DIR=(Join-PATH $env:WORKSPACE TensorRT-8.2.3.0)
$env:ONNXRUNTIME_DIR=(Join-PATH $env:WORKSPACE onnxruntime-win-x64-1.8.1)
$env:CUDNN_DIR=(Join-PATH $env:WORKSPACE cudnn-11.3-v8.2.1.32)

#git clone codebase
# InitMim $codebase_list $env:WORKSPACE $codebase_fullname

#init conda env
conda activate mmdeploy-3.7-$cuda_version

#opencv
$env:path = (Join-PATH $env:OPENCV_DIR bin)+";"+$env:path  

#pplcv
cd $env:WORKSPACE
# git clone https://github.com/openppl-public/ppl.cv.git
cd ppl.cv
# git checkout tags/v0.7.0 -b v0.7.0
$env:PPLCV_DIR = "$pwd"
# mkdir pplcv-build
# cd pplcv-build
# cmake .. -G "Visual Studio 16 2019" -T v142 -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -DPPLCV_USE_CUDA=ON -DPPLCV_USE_MSVC_STATIC_RUNTIME=OFF
# cmake --build . --config Release -- /m
# cmake --install . --config Release
# cd ../..
cd ..

#ONNXRuntime
# pip install onnxruntime==1.8.1
$env:path=(Join-PATH $env:ONNXRUNTIME_DIR lib)+";"+$env:path

#Tensorrt
$env:path =(Join-PATH $env:TENSORRT_DIR lib)+";"+$env:path
# pip install $env:TENSORRT_DIR\python\tensorrt-8.2.3.0-cp37-none-win_amd64.whl
# pip install pycuda

#cudnn
$env:path=(Join-PATH $env:CUDNN_DIR bin)+";"+$env:path

# git clone -b master https://github.com/open-mmlab/mmdeploy.git mmdeploy

# git submodule update --init --recursive
$env:MMDEPLOY_DIR="$pwd/MMdeploy"
cd $env:MMDEPLOY_DIR
rm -r build
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 `
  -DMMDEPLOY_BUILD_SDK=ON `
  -DMMDEPLOY_BUILD_EXAMPLES=ON `
  -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON `
  -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" `
  -DMMDEPLOY_TARGET_BACKENDS="trt;ort" `
  -Dpplcv_DIR="$env:PPLCV_DIR/pplcv-build/install/lib/cmake/ppl" `
  -DTENSORRT_DIR="$env:TENSORRT_DIR" `
  -DONNXRUNTIME_DIR="$env:ONNXRUNTIME_DIR" `
  -DCUDNN_DIR="$env:CUDNN_DIR"
cmake --build . --config Release -- /m
cmake --install . --config Release
cd ..

#add Release Path
$env:path+=";C:\Users\HZJ\Desktop\mmdeploy_windows\MMDeploy\build\bin\Release"

pip install openmim
pip install -r requirements/tests.txt
pip install -r requirements/runtime.txt
pip install -r requirements/build.txt
pip install -v -e .


$date_snap=Get-Date -UFormat "%Y%m%d"
$time_snap=Get-Date -UFormat "%Y%m%d%H%M"
$log_dir=(Join-PATH (Join-PATH "$env:WORKSPACE\regression_log\convert_log" $data_snap) $time_snap)
mkdir $log_dir

$SessionState = [system.management.automation.runspaces.initialsessionstate]::CreateDefault()
$Pool = [runspacefactory]::CreateRunspacePool(1, $max_job_nums, $SessionState, $Host)
$Pool.Open()

$script_block = {
    param(
        [string] $codebase,
        [string] $exec_performance,
        [hashtable] $codebase_fullname_opt,
        [string] $log_dir
    )
    invoke-expression -command "C:\Users\HZJ\Desktop\mmdeploy_windows\MMDeploy\tests\jekins\scripts\win_convert_exec.ps1 $codebase $exec_performance $codebase_fullname_opt *> $log_dir\$codebase.log"
}

$threads = @()

$handles = foreach ($codebase in $codebase_list -split ' ')
{
    $powershell = [powershell]::Create().AddScript($script_block).AddArgument($codebase).AddArgument($exec_performance).AddArgument($codebase_fullname_opt).AddArgument($log_dir)
	  $powershell.RunspacePool = $Pool
	  $powershell.BeginInvoke()
    $threads += $powershell
}

do { 
  $i = 0
  $done = $true
  foreach ($handle in $handles) {
    if ($handle -ne $null) {
  	  if ($handle.IsCompleted) {
        $threads[$i].EndInvoke($handle)
        $threads[$i].Dispose()
        $handles[$i] = $null
      } else {
        $done = $false
      }
    }
    $i++ 
  }
  if (-not $done) { Start-Sleep -Milliseconds 1000 }
} until ($done)
