param($cblist, $eperformance, $rurl, $mbranch, $winconfig ,$mjn)
#set_config

Write-Host "$cblist, $eperformance, $rurl, $mbranch , $winconfig , $mjn"
Write-Host "$pwd"
Copy-Item -Path $pwd/tests/jenkins/conf/$winconfig -Destination $pwd/tests/jenkins/conf/win_tmp.config -Recurse -Force -Verbos

Get-content $pwd/tests/jenkins/conf/win_tmp.config
(Get-content $pwd/tests/jenkins/conf/win_tmp.config) -replace 'codebase_list=.*',"codebase_list=$cblist" | Set-Content $pwd/tests/jenkins/conf/win_tmp.config -Verbos
(Get-content $pwd/tests/jenkins/conf/win_tmp.config) -replace 'exec_performance=.*',"exec_performance=$eperformance" | Set-Content $pwd/tests/jenkins/conf/win_tmp.config -Verbos
(Get-content $pwd/tests/jenkins/conf/win_tmp.config) -replace 'repo_url=.*',"repo_url=$rurl" | Set-Content $pwd/tests/jenkins/conf/win_tmp.config -Verbos
(Get-content $pwd/tests/jenkins/conf/win_tmp.config) -replace 'mmdeploy_branch=.*',"mmdeploy_branch=$mbranch" | Set-Content $pwd/tests/jenkins/conf/win_tmp.config -Verbos
(Get-content $pwd/tests/jenkins/conf/win_tmp.config) -replace 'max_job_nums=.*',"max_job_nums=$mjn" | Set-Content $pwd/tests/jenkins/conf/win_tmp.config -Verbos
#$ConfigPath = './tests/jenkins/conf/win_tmp.config'
#Write-Host "$ConfigPath"
#$content = Get-Content $ConfigPath
#$content.replace('codebase_list=.*', "codebase_list=$cblist") | Set-Content $ConfigPath -Verbos
Get-content $pwd/tests/jenkins/conf/win_tmp.config

# $env:DEPS_DIR="D:\huangzijie\workspace\deps"
# $env:WORKSPACE="D:\huangzijie\workspace"
# $env:OPENCV_DIR=(Join-PATH $env:DEPS_DIR opencv\4.6.0\build\x64\vc15)
# $env:TENSORRT_DIR=(Join-PATH $env:DEPS_DIR TensorRT-8.2.3.0)
# $env:ONNXRUNTIME_DIR=(Join-PATH $env:DEPS_DIR onnxruntime-win-x64-1.8.1)
# $env:CUDNN_DIR=(Join-PATH $env:DEPS_DIR cudnn-11.3-v8.2.1.32)
# $env:PPLCV_DIR=(Join-PATH $env:DEPS_DIR ppl.cv)


$env:DEPS_DIR="D:\DEPS"
$env:WORKSPACE="D:\huangzijie\workspace"
$env:OPENCV_DIR=(Join-PATH $env:DEPS_DIR opencv\4.6.0\build\x64\vc15)
$env:TENSORRT_DIR=(Join-PATH $env:DEPS_DIR tensorrt\TensorRT-8.2.3.0.cuda-11.4.cudnn8.2)
$env:ONNXRUNTIME_DIR=(Join-PATH $env:DEPS_DIR onnxruntime-win-x64-1.8.1)
#$env:CUDNN_DIR=(Join-PATH $env:DEPS_DIR cudnn\cudnn-11.3-v8.2.1.32)
$env:CUDNN_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"
$env:PPLCV_DIR=(Join-PATH $env:DEPS_DIR ppl.cv)



$scriptDir = Split-Path -parent $MyInvocation.MyCommand.Path
Import-Module $scriptDir\utils.psm1

#read configuration file
$config_path = "$pwd\tests\jenkins\conf\win_default.config"
$tmp_config_path = "$pwd\tests\jenkins\conf\win_tmp.config"

if (Test-Path $tmp_config_path){
    $conf = ReadConfig $tmp_config_path

}
else {
    $conf = ReadConfig $config_path

}



if (-not $?) {
    throw "can't load config from $config_path."
}
$env:CUDA_VERSION=$conf.cuda_version
Write-Host "cuda_version=$env:CUDA_VERSION"
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

SwitchCudaVersion $env:CUDA_VERSION
if (-not $?) {
    throw "can't switch cuda version to $env:CUDA_VERSION."
}

if ( $exec_performance -eq "y" ) {
    $exec_performance='--performance'
}else {
    $exec_performance=$null
}
Write-Host "$pwd"
cd ..
$env:JENKINS_WORKSPACE="$pwd"
Write-Host "$pwd"
git clone -b $mmdeploy_branch $repo_url
cd mmdeploy
# git checkout $mmdeploy_branch
# git pull $repo_url
$env:MMDEPLOY_DIR="$pwd"
Write-Host "mmdeploy_dir = $env:MMDEPLOY_DIR"
git submodule update --init --recursive
# Copy-Item -Force -Recurse D:\huangzijie\workspace\tests

$codebase_fullname_opt = @{
    "mmdet" = "mmdetection";
    "mmcls" = "mmclassification";
    "mmdet3d" = "mmdetection3d";
    "mmedit" = "mmediting";
    "mmocr" = "mmocr";
    "mmpose" = "mmpose";
    "mmrotate" = "mmrotate";
    "mmseg" = "mmsegmentation";
    "mmaction" = "mmaction2";
    "mmyolo" = "mmyolo"

}


#git clone codebase
# InitMim $codebase_list $env:WORKSPACE $codebase_fullname

#init conda env

# $codebase_list = "mmdet", "mmcls"

foreach ($codebase in $codebase_list -split ' ') {
    Write-Host "conda activate mmdeploy-3.7-$env:CUDA_VERSION-$codebase"
    conda activate mmdeploy-3.7-$env:CUDA_VERSION-$codebase

    #opencv
    $env:path = (Join-PATH $env:DEPS_DIR opencv\4.6.0\build)+";"+$env:path
    $env:path = (Join-PATH $env:OPENCV_DIR bin)+";"+$env:path
    $env:path = (Join-PATH $env:OPENCV_DIR lib)+";"+$env:path
    #ONNXRuntime
    # pip install onnxruntime==1.8.1
    $env:path=(Join-PATH $env:ONNXRUNTIME_DIR lib)+";"+$env:path

    #Tensorrt
    $env:path =(Join-PATH $env:TENSORRT_DIR lib)+";"+$env:path
    #cudnn
    $env:path=(Join-PATH $env:CUDNN_DIR bin)+";"+$env:path
    New-Item -Path $env:MMDEPLOY_DIR\data -ItemType SymbolicLink -Value  D:\huangzijie\workspace\data
    net use \\10.1.52.36\public\benchmark 123456 /user:zhengshaofeng
    mkdir build
    cd build
    cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 `
        -DMMDEPLOY_BUILD_SDK=ON `
        -DMMDEPLOY_BUILD_EXAMPLES=ON `
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON `
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" `
        -DMMDEPLOY_TARGET_BACKENDS="trt;ort" `
        -Dpplcv_DIR="$env:PPLCV_DIR\pplcv-build\install\lib\cmake\ppl" `
        -DTENSORRT_DIR="$env:TENSORRT_DIR" `
        -DONNXRUNTIME_DIR="$env:ONNXRUNTIME_DIR" `
        -DCUDNN_DIR="$env:CUDNN_DIR"
    cmake --build . --config Release -- /m
    cmake --install . --config Release
    cd ..

    add Release Path
    $env:path+=";$env:MMDEPLOY_DIR\build\bin\Release"

    pip install openmim
    pip install -r requirements/tests.txt
    pip install -r requirements/runtime.txt
    pip install -r requirements/build.txt
    pip install -v -e .
    $date_snap=Get-Date -UFormat "%Y%m%d"
    $time_snap=Get-Date -UFormat "%Y%m%d%H%M"
# $log_dir=(Join-PATH (Join-PATH "$env:WORKSPACE\regression_log\convert_log" $data_snap) $time_snap)
    $log_dir = (Join-PATH (Join-Path "$env:WORKSPACE\mmdeploy_regression_working_dir\$codebase\$env:CUDA_VERSION" $data_snap) $time_snap)
    Write-Host "log_dir = $log_dir"
    mkdir $log_dir

    $SessionState = [system.management.automation.runspaces.initialsessionstate]::CreateDefault()
    $Pool = [runspacefactory]::CreateRunspacePool(1, $max_job_nums, $SessionState, $Host)
    $Pool.Open()

    $script_block = {
        param(
        [string] $codebase,
        [string] $exec_performance,
        [string] $codebase_fullname,
        [string] $log_dir,
        [string] $scriptDir,
        [string] $mmdeploy_branch
    )
    Write-Host "$scriptDir\win_convert_exec.ps1 $codebase $exec_performance $codebase_fullname $mmdeploy_branch *> $log_dir\$codebase.txt"
    invoke-expression -command "$scriptDir\win_convert_exec.ps1 $codebase $exec_performance $codebase_fullname $mmdeploy_branch *> $log_dir\$codebase.txt"
}

}

# conda activate mmdeploy-3.7-$env:CUDA_VERSION
# Write-Host "conda activate mmdeploy-3.7-$env:CUDA_VERSION"


#pplcv
# cd $env:WORKSPACE
# git clone https://github.com/openppl-public/ppl.cv.git
# cd ppl.cv
# git checkout tags/v0.7.0 -b v0.7.0
# $env:PPLCV_DIR = "$pwd"
# mkdir pplcv-build
# cd pplcv-build
# cmake .. -G "Visual Studio 16 2019" -T v142 -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install -DPPLCV_USE_CUDA=ON -DPPLCV_USE_MSVC_STATIC_RUNTIME=OFF
# cmake --build . --config Release -- /m
# cmake --install . --config Release
# cd ../..
# cd ..


# pip install $env:TENSORRT_DIR\python\tensorrt-8.2.3.0-cp37-none-win_amd64.whl
# pip install pycuda



#git clone -b $mmdeploy_branch  https://github.com/open-mmlab/mmdeploy.git Tmp
#git submodule update --init --recursive
#Remove-Item -Force -Recurse .\Tmp\.git
#Copy-Item -Path $pwd\Tmp\* -Recurse $pwd\ -Force
#rm -r Tmp
#
# New-Item -Path $env:MMDEPLOY_DIR\data -ItemType SymbolicLink -Value  D:\huangzijie\workspace\data
# net use \\10.1.52.36\public\benchmark 123456 /user:zhengshaofeng

# New-Item -Path $env:MMDEPLOY_DIR\data -ItemType SymbolicLink -Value  \\10.1.52.36\public\benchmark

# New-Item -ItemType SymbolicLink -Path "D:\huangzijie\workspace\mmdeploy_win\mmdeploy\data" -Target "Z:\"


# mkdir build
# cd build
# cmake .. -G "Visual Studio 16 2019" -A x64 -T v142 `
#   -DMMDEPLOY_BUILD_SDK=ON `
#   -DMMDEPLOY_BUILD_EXAMPLES=ON `
#   -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON `
#   -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" `
#   -DMMDEPLOY_TARGET_BACKENDS="trt;ort" `
#   -Dpplcv_DIR="$env:PPLCV_DIR\pplcv-build\install\lib\cmake\ppl" `
#   -DTENSORRT_DIR="$env:TENSORRT_DIR" `
#   -DONNXRUNTIME_DIR="$env:ONNXRUNTIME_DIR" `
#   -DCUDNN_DIR="$env:CUDNN_DIR"
# cmake --build . --config Release -- /m
# cmake --install . --config Release
# cd ..

#add Release Path
# $env:path+=";$env:MMDEPLOY_DIR\build\bin\Release"

# pip install openmim
# pip install -r requirements/tests.txt
# pip install -r requirements/runtime.txt
# pip install -r requirements/build.txt
# pip install -v -e .

# $date_snap=Get-Date -UFormat "%Y%m%d"
# $time_snap=Get-Date -UFormat "%Y%m%d%H%M"
# # $log_dir=(Join-PATH (Join-PATH "$env:WORKSPACE\regression_log\convert_log" $data_snap) $time_snap)
# $log_dir = (Join-PATH (Join-Path "$env:WORKSPACE\mmdeploy_regression_working_dir\$codebase\$env:CUDA_VERSION" $data_snap) $time_snap)
# mkdir $log_dir
#
# $SessionState = [system.management.automation.runspaces.initialsessionstate]::CreateDefault()
# $Pool = [runspacefactory]::CreateRunspacePool(1, $max_job_nums, $SessionState, $Host)
# $Pool.Open()
#
# $script_block = {
#     param(
#         [string] $codebase,
#         [string] $exec_performance,
#         [string] $codebase_fullname,
#         [string] $log_dir,
#         [string] $scriptDir,
#         [string] $mmdeploy_branch
#     )
#     Write-Host "$scriptDir\win_convert_exec.ps1 $codebase $exec_performance $codebase_fullname $mmdeploy_branch *> $log_dir\$codebase.txt"
#     invoke-expression -command "$scriptDir\win_convert_exec.ps1 $codebase $exec_performance $codebase_fullname $mmdeploy_branch *> $log_dir\$codebase.txt"
# }

$threads = @()

$handles = foreach ($codebase in $codebase_list -split ' ')
{
    $codebase_fullname = $codebase_fullname_opt.([string]$codebase)
    $powershell = [powershell]::Create().AddScript($script_block).AddArgument($codebase).AddArgument($exec_performance).AddArgument($codebase_fullname).AddArgument($log_dir).AddArgument($scriptDir).AddArgument($mmdeploy_branch)
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
