<#
.SYNOPSIS
A helper script to test tools of MMDeploy Converter on windows.

.Description
    -Backend: support ort, trt
    -Device: support cpu, cuda, cuda:0

.EXAMPLE
PS> .github/scripts/linux/test_full_pipeline.sh -Backend ort -Device cpu
#>

param(
    [Parameter(Mandatory = $true)]
    [string] $Backend,
    [string] $Device
)

$MMDeploy_DIR="$PSScriptRoot\..\..\.."
Set-Location $MMDeploy_DIR

$work_dir="work_dir"
New-Item -Path $work_dir, .\data -ItemType Directory -Force
$model_cfg="$work_dir\resnet18_8xb32_in1k.py"
$checkpoint="$work_dir\resnet18_8xb32_in1k_20210831-fbbb1da6.pth"
$sdk_cfg="configs\mmpretrain\classification_sdk_dynamic.py"
$input_img="tests\data\tiger.jpeg"

python -m mim download mmpretrain --config resnet18_8xb32_in1k --dest $work_dir

if ($Backend -eq "ort") {
    $deploy_cfg="configs\mmpretrain\classification_onnxruntime_dynamic.py"
    $model="$work_dir\end2end.onnx"
} elseif ($Backend -eq "trt") {
    $deploy_cfg="configs\mmpretrain\classification_tensorrt-fp16_dynamic-224x224-224x224.py"
    $model="$work_dir\end2end.engine"
} else {
    Write-Host "Unsupported Backend=$Backend"
    Exit
}

Write-Host "--------------------------------------"
Write-Host "deploy_cfg=$deploy_cfg"
Write-Host "$model_cfg=$model_cfg"
Write-Host "$checkpoint=$checkpoint"
Write-Host "device=$device"
Write-Host "--------------------------------------"

python tools\deploy.py `
  $deploy_cfg `
  $model_cfg `
  $checkpoint `
  $input_img `
  --device $device `
  --work-dir $work_dir `
  --dump-info

# prepare dataset
Invoke-WebRequest -Uri https://github.com/open-mmlab/mmdeploy/releases/download/v0.1.0/imagenet-val100.zip -OutFile $pwd\data\imagenet-val100.zip
Add-Type -AssemblyName System.IO.Compression.FileSystem ; [System.IO.Compression.ZipFile]::ExtractToDirectory("$pwd\data\imagenet-val100.zip", "$pwd\data\")

Write-Host "Running test with ort"
python tools\test.py `
  $deploy_cfg `
  $model_cfg `
  --model $model `
  --device $device `
  --log2file $work_dir\test_ort.log `
  --speed-test `
  --log-interval 50 `
  --warmup 20 `
  --batch-size 8


Write-Host "Prepare dataset"
# change topk for test
$src_topk='"topk": 5'
$dst_topk='"topk": 1000'
$src_pipeline_file="$work_dir\pipeline.json"
$tmp_pipeline_file="$work_dir\pipeline_tmp.json"
Move-Item $src_pipeline_file $tmp_pipeline_file -force
(Get-Content -Path $tmp_pipeline_file) -replace $src_topk, $dst_topk | Add-Content -Path $src_pipeline_file

Write-Host "test sdk model"

python tools\test.py `
  $sdk_cfg `
  $model_cfg `
  --model $work_dir `
  --device $device `
  --log2file $work_dir\test_sdk.log `
  --speed-test `
  --log-interval 50 `
  --warmup 20 `
  --batch-size 8`

# test profiler
Write-Host "Profile sdk model"
python tools\profiler.py `
  $sdk_cfg `
  $model_cfg `
  .\data `
  --model $work_dir `
  --device $device `
  --batch-size 8 `
  --shape 224x224

# remove temp data
Remove-Item -Path "$work_dir" -Force -Recurse
Write-Host "All done"
