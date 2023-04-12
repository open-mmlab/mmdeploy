param(
    $codebase,
    $exec_performance,
    $codebase_fullname,
    $mmdeploy_branch
)
$scriptDir = Split-Path -parent $MyInvocation.MyCommand.Path
$confDir=(Join-PATH $env:JENKINS_WORKSPACE master@2\tests\jenkins\conf)
Import-Module $scriptDir\utils.psm1
$json_v1 = Get-Content -Path "$confDir\requirementV1.0.json" -Raw  |  ConvertFrom-Json
$json_v2 = Get-Content -Path "$confDir\requirementV2.0.json" -Raw  |  ConvertFrom-Json
cd $env:MMDEPLOY_DIR
conda activate mmdeploy-3.7-cu113
Write-Host "exec_path: $pwd"
Write-Host "mim install $codebase"
Write-Host "codebase_fullname = $codebase_fullname"
Write-Host "exec_performance = $exec_performance"
Write-Host "mmdeploy_branch = $mmdeploy_branch"
$codebase_path = (Join-Path $env:JENKINS_WORKSPACE $codebase_fullname)
Write-Host "codebase_path = $codebase_path"
$log_dir = (Join-Path $env:WORKSPACE "mmdeploy_regression_working_dir\$codebase\$env:CUDA_VERSION")
Write-Host "log_dir = $log_dir"
InitMim $codebase $codebase_fullname $mmdeploy_branch
pip uninstall mmcv-full -y
pip uninstall mmcv -y
pip uninstall $codebase -y

mim list
pip install -v $codebase_path
mim list
mim install $codebase
if ($mmdeploy_branch -eq "master"){
    if ($json_v1.PSObject.Properties.Name -contains $codebase)
    {
        $mmcv = $json_v1.$codebase.mmcv
        mim install $mmcv
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
        mim install $mmcv
    }
    else
    {
        Write-Host "$codebase not found in requirementV2.0.json file"
    }
}
mim list
Write-Host "$pwd"
#Invoke-Expression -Command "python3 ./tools/regression_test.py --codebase $codebase --device cuda:0 --backends tensorrt onnxruntime --work-dir $log_dir  $exec_performance"
# python3 ./tools/regression_test.py --codebase $codebase --device cuda:0 --backends tensorrt onnxruntime --work-dir $log_dir  $exec_performance
python $pwd/tools/regression_test.py `
    --codebase $codebase `
    --device cuda:0 `
    --backends tensorrt onnxruntime `
    --work-dir $log_dir  `
    $exec_performance

if (-not $?) {
    throw "regression_test failed"
}