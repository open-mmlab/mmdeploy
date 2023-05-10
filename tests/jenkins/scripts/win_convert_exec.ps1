param(
    $codebase,
    $exec_performance,
    $codebase_fullname,
    $mmdeploy_branch
)


$scriptDir = Split-Path -parent $MyInvocation.MyCommand.Path
$confDir=(Join-PATH $env:JENKINS_WORKSPACE master\tests\jenkins\conf)
Import-Module $scriptDir\utils.psm1
$json_v1 = Get-Content -Path "$confDir\requirementV1.0.json" -Raw  |  ConvertFrom-Json
$json_v2 = Get-Content -Path "$confDir\requirementV2.0.json" -Raw  |  ConvertFrom-Json
cd $env:MMDEPLOY_DIR
conda activate mmdeploy-3.7-cu113-$codebase
Write-Host "conda activate mmdeploy-3.7-cu113-$codebase"
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
# Write-Host "build end"
# $env:MMDEPLOY_DIR="$pwd"
# $env:path+=";$env:MMDEPLOY_DIR\build\bin\Release"
# Write-Host "start to pip requirements"
# pip install openmim
# pip install -r requirements/tests.txt
# pip install -r requirements/runtime.txt
# pip install -r requirements/build.txt
# pip install -v -e .
Write-Host "exec_path: $pwd"
Write-Host "mim install $codebase"
Write-Host "codebase_fullname = $codebase_fullname"
Write-Host "exec_performance = $exec_performance"
Write-Host "mmdeploy_branch = $mmdeploy_branch"
$codebase_path = (Join-Path $env:JENKINS_WORKSPACE $codebase_fullname)
Write-Host "codebase_path = $codebase_path"
$date_snap=Get-Date -UFormat "%Y%m%d"
$time_snap=Get-Date -UFormat "%Y%m%d%H%M"
$log_dir = (Join-PATH(Join-Path $env:WORKSPACE "mmdeploy_regression_working_dir\$codebase\$env:CUDA_VERSION"$data_snap)$time_snap)
Write-Host "log_dir = $log_dir"
InitMim $codebase $codebase_fullname $mmdeploy_branch
python -m pip uninstall mmcv-full -y
python -m pip uninstall mmcv -y
python -m pip uninstall $codebase -y

mim list
python -m pip install -v $codebase_path
mim list
python -m mim install $codebase
if ($mmdeploy_branch -eq "master"){
    if ($json_v1.PSObject.Properties.Name -contains $codebase)
    {
        $mmcv = $json_v1.$codebase.mmcv
        python -m mim install $mmcv
    }
    else
    {
        Write-Host "$codebase not found in requirementV1.0.json file"
    }
}
elseif ($mmdeploy_branch -eq "main"){
    if ($json_v2.PSObject.Properties.Name -contains $codebase)
    {
        $mmcv = $json_v2.$codebase.mmcv
        python -m mim install $mmcv
    }
    else
    {
        Write-Host "$codebase not found in requirementV2.0.json file"
    }
}
mim list
python -m mim list
Write-Host "$pwd"
#Invoke-Expression -Command "python3 ./tools/regression_test.py --codebase $codebase --device cuda:0 --backends tensorrt onnxruntime --work-dir $log_dir  $exec_performance"
# python3 ./tools/regression_test.py --codebase $codebase --device cuda:0 --backends tensorrt onnxruntime --work-dir $log_dir  $exec_performance
python $pwd/tools/regression_test.py `
    --codebase $codebase `
    --device cuda:0 `
    --backends tensorrt onnxruntime `
    --work-dir $log_dir  `
    $exec_performance

cd ..
python ./master/tests/jenkins/scripts/check_results.py `
    $log_dir `
    --regression-dir $log_dir
if (-not $?) {
    throw "regression_test failed"
}
