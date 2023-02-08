<#
.SYNOPSIS
A helper script to help set env and download mmdeploy deps on windows.


.Description
This script have four usages:
    1) .\mmdeploy_init.ps1 -Action Download
    2) .\mmdeploy_init.ps1 -Action SetupEnv
    3) .\mmdeploy_init.ps1 -Action Build
    4) .\mmdeploy_init.ps1 -Action Install -Dir path\to\exe

.EXAMPLE
PS> .\mmdeploy_init.ps1
#>

param(
    [Parameter(Mandatory = $true)]
    [string] $Action,
    [string] $Dir = "3rdparty"
)

# BEGIN Model converter && SDK config

##### [common] #####

# build type
# possible values:
# - "Release": release build
# - "Debug": debug build
$CMAKE_BUILD_TYPE = "Release"

# target inference engines to support
# possible values:
# - "trt": tensorrt
# - "ort": onnxruntime
# multiple backends should be seperated by ; like "trt;ort"
$MMDEPLOY_TARGET_BACKENDS = "trt"

# onnxruntime config, valid if "ort" in MMDEPLOY_TARGET_BACKENDS
#
# onnxruntime package type
# possible values:
# - "cpu": build with onnxruntime
# - "cuda": build with onnxruntime-gpu
$onnxruntimeDevice = "cpu"
#
# onnxruntime package version
# possible values:
# - ["1.8.1"]
$onnxruntimeVersion = "1.8.1"

# tensorrt config, valid if "trt" in MMDEPLOY_TARGET_BACKENDS
#
# tensorrt version
# possible values:
# - ["8.2.3", "8.2.4", "8.2.5", "8.4.1", "8.4.2", "8.4.3", "8.5.1", "8.5.2"]
$tensorrtVersion = "8.2.3"
#
# cudnn version
# possible values:
# - ["8.2.1", "8.4.1", "8.6.0"]
$cudnnVersion = "8.2.1"

# cuda config
# possible values:
# - ["10.2", "11.x"]
$cudaVersion = "11.x"


##### [SDK] #####

# whether build mmdeploy sdk
# possible values:
# - ["ON", "OFF"]
$MMDEPLOY_BUILD_SDK = "ON"

# whether build sdk python api
# possible values:
# - ["ON", "OFF"]
$MMDEPLOY_BUILD_SDK_PYTHON_API = "OFF"

# whether build csharp api
# possible values:
# - ["ON", "OFF"]
$MMDEPLOY_BUILD_SDK_CSHARP_API = "OFF"

# whether build examples
# possible values:
# - ["ON", "OFF"]
$MMDEPLOY_BUILD_EXAMPLES = "OFF"

# target devices to support
# possible values:
# - "cpu": cpu
# - "cuda": gpu
# - "acl": atlas
# multiple backends should be seperated by ; like "cpu;cuda"
$MMDEPLOY_TARGET_DEVICES = "cpu;cuda"


# opencv config, will first consider $opencvCustomFolder, then $opencvVersion
#
# prebuilt opencv version.
# possible values:
# - ["4.5.5"]
$opencvVersion = "4.5.5"
#
# use custom opencv, the folder should contain OpenCVConfig.cmake
$opencvCustomFolder = ""

# pplcv root dir, if specified, will use user defined pplcv
# thr root dir should contains lib and include folder.
$pplcvCustomFolder = "C:\Deps\ppl.cv\pplcv-build\install"

# codebase config
# possible values:
# - ["mmcls", "mmdet", "mmseg", "mmpose", "mmocr", "mmedit", "mmrotate", "mmaction"]
# multiple backends should be seperated by ; like "mmcls;mmdet"
$MMDEPLOY_CODEBASES = "all"


# END Model converter && SDK config

$ErrorActionPreference = 'Stop'

$EnvVars = @{}
$EnvPath = [System.Collections.ArrayList]@()


$envFile = "env.txt"

Import-Module BitsTransfer

function Show-HelpMessage {
    param ()

    Write-Host "usage: .\mmdeploy_init.ps1 command ..."
}

class DownloadItem {
    [string]$Url
    [string]$Md5
    DownloadItem([string]$Url, [string]$Md5) {
        $this.Url = $Url
        $this.Md5 = $Md5
    }
}

function Start-DownloadWithRetry {
    param (
        [string] $Url,
        [string] $Name,
        [string] $Md5 = "",
        [string] $SaveDir = "${env:Temp}",
        [int] $Retries = 3
    )

    Write-Host "Download $Name ..."
    $fileName = [IO.Path]::GetFileName($Url)
    $filePath = Join-Path -Path $SaveDir -ChildPath $fileName
    $downloadStartTime = Get-Date

    while ($Retries -gt 0) {
        if (Test-Path $filePath) {
            $md5_val = (Get-FileHash $filePath -Algorithm MD5).Hash
            if ($md5_val -eq $Md5) {
                Write-Host "Using cached $filePath"
                break
            }
        }
        try {
            $downloadAttemptStartTime = Get-Date
            Write-Host "Downloading package from: $Url to path $filePath ."
            Start-BitsTransfer $Url $filePath
            # (New-Object System.Net.WebClient).DownloadFile($Url, $filePath)
            # Invoke-WebRequest $Url -OutFile $filePath
            $downloadCompleteTime = [math]::Round(($(Get-Date) - $downloadStartTime).TotalSeconds, 2)
            Write-Host "Package $Name downloaded successfully in $downloadCompleteTime seconds"
            break
        }
        catch {
            $attemptTime = [math]::Round(($(Get-Date) - $downloadAttemptStartTime).TotalSeconds, 2)
            Write-Warning "There is an error encounterd after $attemptTime seconds during package downloading:`n $_"
            $Retries--

            if ($Retries -eq 0) {
                Write-Error "File can't be downloaded. Please try later or check that file exists by url: $Url"
                exit 1
            }

            Write-Warning "Waiting 5 seconds before retrying. Retries left: $Retries"
            Start-Sleep -Seconds 5
        }
    }
    return $filePath
}

function Get-Opencv {
    param (
        $Version,
        $CustomFolder = ""
    )

    if ($CustomFolder -ne "") {
        Write-Host "Using custom opencv $CustomFolder"
        $opencvBase = [IO.Path]::Combine($CustomFolder, "x64")
        if ((Get-ChildItem $opencvBase -Filter "vc*").Length -gt 1) {
            $vcx = (Get-ChildItem $opencvBase -Filter "vc*").Name[-1]
        }
        else {
            $vcx = (Get-ChildItem $opencvBase -Filter "vc*").Name
        }
        # set variable
        $EnvVars["OpenCV_DIR"] = [IO.Path]::Combine($opencvBase, $vcx, "lib")
        [void]$EnvPath.Add([IO.Path]::Combine($opencvBase, $vcx, "bin"))
        return;
    }

    $opencvUrls = @{
        "4.5.5" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/opencv/opencv-win-amd64-4.5.5-vc16.zip",
            "e1a744108105fb20e3c8371d58b38c1a"
        );
    }
    $item = $opencvUrls.$Version
    $fileSrc = Start-DownloadWithRetry -Url $item.Url -Name "opencv" -Md5 $item.Md5
    $fileDst = [IO.Path]::Combine($pwd, $Dir, 'opencv')
    # extract
    Write-Host "Extract file $fileSrc to $fileDst"
    Expand-Archive -Path $fileSrc -DestinationPath $fileDst -Force
    Write-Host "Extract file $fileSrc done"
    # set variable
    $subFolder = (Get-ChildItem $fileDst).Name
    $opencvBase = [IO.Path]::Combine($fileDst, $subFolder, "x64")
    if ((Get-ChildItem $opencvBase -Filter "vc*").Length -gt 1) {
        $vcx = (Get-ChildItem $opencvBase -Filter "vc*").Name[-1]
    }
    else {
        $vcx = (Get-ChildItem $opencvBase -Filter "vc*").Name
    }
    $EnvVars["OpenCV_DIR"] = [IO.Path]::Combine($opencvBase, $vcx, "lib")
    [void]$EnvPath.Add([IO.Path]::Combine($opencvBase, $vcx, "bin"))
}

function Get-Onnxruntime {
    param (
        [string]$Device,
        [string]$Version
    )

    $sys_onnxruntime = @("C:\Windows\System32\onnxruntime.dll", "C:\Windows\SysWOW64\onnxruntime.dll")
    foreach ($path in $sys_onnxruntime) {
        if (Test-Path $path) {
            Write-Warning "`nFound onnxruntime.dll in $path, it will conflict with downloaded version of onnxruntime, please rename it"
        }
    }

    $ortCpuUrls = @{
        "1.8.1" = [DownloadItem]::new(
            "https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-win-x64-1.8.1.zip",
            "dabe12f86a4a37caa4feec16ef16ade6"
        );
    }
    $ortCudaUrls = @{
        "1.8.1" = [DownloadItem]::new(
            "https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-win-gpu-x64-1.8.1.zip",
            "495442f6f124597635af3cfa96213122"
        )
    }
    $ortUrls = @{
        "cpu"  = $ortCpuUrls;
        "cuda" = $ortCudaUrls
    }
    $item = $ortUrls.$Device.$Version
    $fileSrc = Start-DownloadWithRetry -Url $item.Url -Name "ONNX Runtime" -Md5 $item.Md5
    $fileDst = [IO.Path]::Combine($pwd, $Dir, 'onnxruntime')
    # extract
    Write-Host "Extract file $fileSrc to $fileDst"
    Expand-Archive -Path $fileSrc -DestinationPath $fileDst -Force
    Write-Host "Extract file $fileSrc done"
    # set variable
    $subFolder = (Get-ChildItem $fileDst).Name
    $onnxruntimeDir = [IO.Path]::Combine($fileDst, $subFolder)
    $EnvVars["ONNXRUNTIME_DIR"] = $onnxruntimeDir
    [void]$EnvPath.Add([IO.Path]::Combine($onnxruntimeDir, "lib"))
}



function Get-Tensorrt {
    param(
        [ValidateSet("10.2", "11.x")]
        [string] $CudaVerion,
        [ValidateSet("8.2.3", "8.2.4", "8.2.5", "8.4.1", "8.4.2", "8.4.3", "8.5.1", "8.5.2")]
        [string] $Version
    )
    # download
    $tensorrtCu102Urls = @{
        "8.2.3" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.2.3.0.Windows10.x86_64.cuda-10.2.cudnn8.2.zip",
            "e6b5a412101f8dc823eaf11bfb968376");
        "8.2.4" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.2.4.2.Windows10.x86_64.cuda-10.2.cudnn8.2.zip",
            "05d9d817ae77da394fd80ad2df729329");
        "8.2.5" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.2.5.1.Windows10.x86_64.cuda-10.2.cudnn8.2.zip",
            "7274036d6a2320e0c1e39d5ea0886611");
        "8.4.1" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.4.1.5.Windows10.x86_64.cuda-10.2.cudnn8.4.zip",
            "7059ce9ebcf17ee8d0fff29d3f2d9c00");
        "8.4.2" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.4.2.4.Windows10.x86_64.cuda-10.2.cudnn8.4.zip",
            "8ade017b1075fcdb1a395622f1d3ce7d");
        "8.4.3" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.4.3.1.Windows10.x86_64.cuda-10.2.cudnn8.4.zip",
            "55044d4f927f28bfd9b45073f7d15a84");
        "8.5.1" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.5.1.7.Windows10.x86_64.cuda-10.2.cudnn8.6.zip",
            "e00d666292625781e8e5715c9687264c");
        "8.5.2" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.5.2.2.Windows10.x86_64.cuda-10.2.cudnn8.6.zip",
            "d52692c8f615ec2142c968214185451f");
    }
    $tensorrtCu11xUrls = @{
        "8.2.3" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.2.3.0.Windows10.x86_64.cuda-11.4.cudnn8.2.zip",
            "3364ad175781746a0e7359d2ed4b2287");
        "8.2.4" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.2.4.2.Windows10.x86_64.cuda-11.4.cudnn8.2.zip",
            "ac7252b92cb76084f875554a5f8f759a");
        "8.2.5" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.2.5.1.Windows10.x86_64.cuda-11.4.cudnn8.2.zip",
            "505e35eed6dc9ca4784199d0dc3b00e6");
        "8.4.1" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.4.1.5.Windows10.x86_64.cuda-11.6.cudnn8.4.zip",
            "8b5afa6d9abea36d4cfe3df831faad14");
        "8.4.2" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.4.2.4.Windows10.x86_64.cuda-11.6.cudnn8.4.zip",
            "0b2691dfdc5e40d97f3ca9249fc8f9b1");
        "8.4.3" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.4.3.1.Windows10.x86_64.cuda-11.6.cudnn8.4.zip",
            "125b987f6571cf8b43528f0a26f46593");
        "8.5.1" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.5.1.7.Windows10.x86_64.cuda-11.8.cudnn8.6.zip",
            "775667c49cd02f8fd14c305590978a97");
        "8.5.2" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/tensorrt/TensorRT-8.5.2.2.Windows10.x86_64.cuda-11.8.cudnn8.6.zip",
            "b67ba529ad54338fee6e9d282c5f272c");
    }
    $tensorrtUrls = @{
        "10.2" = $tensorrtCu102Urls;
        "11.x" = $tensorrtCu11xUrls
    }
    $item = $tensorrtUrls.$CudaVerion.$Version
    $fileSrc = Start-DownloadWithRetry -Url $item.Url -Name "TensorRT" -Md5 $item.Md5
    $fileDst = [IO.Path]::Combine($pwd, $Dir, 'tensorrt')
    # extract
    Write-Host "Extract file $fileSrc to $fileDst"
    Expand-Archive -Path $fileSrc -DestinationPath $fileDst -Force
    Write-Host "Extract file $fileSrc done"
    # set variable
    $subFolder = (Get-ChildItem $fileDst).Name
    $tensorrtDir = [IO.Path]::Combine($fileDst, $subFolder)
    $EnvVars["TENSORRT_DIR"] = $tensorrtDir
    [void]$EnvPath.Add([IO.Path]::Combine($tensorrtDir, 'lib'))
}

function Get-Cudnn {
    param (
        [ValidateSet("10.2", "11.x")]
        [string] $CudaVerion,
        [ValidateSet("8.2.1", "8.4.1", "8.6.0")]
        [string] $Version
    )
    # download
    $cudnnCu102Urls = @{
        "8.2.1" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/cudnn/cudnn-10.2-windows10-x64-v8.2.1.32.zip",
            "14ce30ae33b5e2a7d4f1e124a251219a");
        "8.4.1" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/cudnn/cudnn-windows-x86_64-8.4.1.50_cuda10.2-archive.zip",
            "5cccd75f8300b70e83800a4368ebce1f");
        "8.6.0" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/cudnn/cudnn-windows-x86_64-8.6.0.163_cuda10-archive.zip",
            "fb84f9f990373e239ddb9515c30c754e")
    }
    $cudnnCu11xUrls = @{
        "8.2.1" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/cudnn/cudnn-11.3-windows-x64-v8.2.1.32.zip",
            "1f6616595f397fe02a64f4d828e6eeb6");
        "8.4.1" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/cudnn/cudnn-windows-x86_64-8.4.1.50_cuda11.6-archive.zip",
            "d770cf6922dc76d82bdc08af3cd19603");
        "8.6.0" = [DownloadItem]::new(
            "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/cudnn/cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip",
            "55f0fc87255861ab10b0b796d271256c")
    }
    $cudnnUrls = @{
        "10.2" = $cudnnCu102Urls;
        "11.x" = $cudnnCu11xUrls
    }
    $item = $cudnnUrls.$CudaVerion.$Version
    $fileSrc = Start-DownloadWithRetry -Url $item.Url -Name "cuDNN" -Md5 $item.Md5
    $fileDst = [IO.Path]::Combine($pwd, $Dir, "cudnn")
    # extract
    Write-Host "Extract file $fileSrc to $fileDst"
    Expand-Archive -Path $fileSrc -DestinationPath $fileDst -Force
    Write-Host "Extract file $fileSrc done"
    # set variable
    $EnvVars["CUDNN_DIR"] = $fileDst
    [void]$EnvPath.Add([IO.Path]::Combine($fileDst, "cuda", 'bin'))
}

function Get-Pplcv {
    param (
        [ValidateSet("10.2", "11.x")]
        [string] $CudaVerion,
        [string] $CustomFolder = ""
    )
    if ($CustomFolder -ne "") {
        Write-Host "Using custom pplcv $CustomFolder"
        # set variable
        $pplcvDir = [IO.Path]::Combine($CustomFolder, "lib", "cmake", "ppl")
        $EnvVars["pplcv_DIR"] = $pplcvDir
        return;
    }
    # download
    $pplcvCu102url = [DownloadItem]::new(
        "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/pplcv/pplcv-0.6.2.windows10.x86_64.cuda-10.2.zip",
        "800386f8ffbfeb5d0bf6e056bdd2a98a");
    $pplcvCu11xurl = [DownloadItem]::new(
        "https://github.com/irexyc/mmdeploy-ci-resource/releases/download/pplcv/pplcv-0.6.2.windows10.x86_64.cuda-11.x.zip",
        "26d028c4f19a86d6feb28614caa461ef");
    $pplcvUrls = @{
        "10.2" = $pplcvCu102url;
        "11.x" = $pplcvCu11xurl
    }
    $item = [DownloadItem]($pplcvUrls.$CudaVerion)
    $fileSrc = Start-DownloadWithRetry -Url $item.Url -Name "ppl.cv" -Md5 $item.Md5
    $fileDst = [IO.Path]::Combine($pwd, $Dir, 'pplcv')
    # extract
    Write-Host "Extract file $fileSrc to $fileDst"
    Expand-Archive -Path $fileSrc -DestinationPath $fileDst -Force
    Write-Host "Extract file $fileSrc done"
    # set varialbe
    $pplcvDir = [IO.Path]::Combine($fileDst, "lib", "cmake", "ppl")
    $EnvVars["pplcv_DIR"] = $pplcvDir
}

function Write-EnvFile {
    param ()

    "[environment variables]" | Out-File -FilePath $envFile -Append
    foreach ($key in $EnvVars.Keys) {
        "$key $($EnvVars.$key)" | Out-File -FilePath $envFile -Append
    }

    "[environment path]" | Out-File -FilePath $envFile -Append
    foreach ($val in $EnvPath) {
        "$val" | Out-File -FilePath $envFile -Append
    }
}

function Get-Dependences {
    param ()

    if (Test-Path $envFile) {
        remove-item $envFile
    }

    # backend
    if ($MMDEPLOY_TARGET_BACKENDS.Contains("trt")) {
        Write-Host "Using tensorrt and cudnn for cuda $cudaVersion"

        Get-Tensorrt -CudaVerion $cudaVersion -Version $tensorrtVersion
        Get-Cudnn -CudaVerion $cudaVersion -Version $cudnnVersion
    }
    if ($MMDEPLOY_TARGET_BACKENDS.Contains("ort")) {
        Get-Onnxruntime -Device $onnxruntimeDevice -Version $onnxruntimeVersion
    }

    # sdk
    if ($MMDEPLOY_BUILD_SDK -ne "ON") {
        Write-Host "Skip build mmdeploy sdk"
        return
    }

    # pplcv if use cuda device
    if ($MMDEPLOY_TARGET_DEVICES.Contains("cuda")) {
        Get-Pplcv -CudaVerion $cudaVersion -CustomFolder $pplcvCustomFolder
    }

    # opencv
    Get-Opencv -Version $opencvVersion -CustomFolder $opencvCustomFolder
}

function Set-Environment {
    # [environment variables]
    # [environment path]
    param ()
    if (-Not (Test-Path $envFile)) {
        Write-Error "Can't find env.txt in $pwd, Please first download depencencies"
    }
    $env_content = Get-Content $envFile

    Write-Host "Setup up environment variables ..."
    $i = 1
    for (; $i -lt $env_content.Length; $i++) {
        if ($env_content[$i][0] -eq '[') {
            $i++;
            break;
        }
        $kv = $env_content[$i].Split(' ')
        [System.Environment]::SetEnvironmentVariable($kv[0], $kv[1]);
        $EnvVars[$kv[0]] = $kv[1]
    }
    Get-ChildItem Env:

    # print environment variable
    Write-Host "`nSetup up environment path ..."
    $path_val = [Environment]::GetEnvironmentVariable("PATH")
    $path_list = $path_val.Split( [IO.Path]::PathSeparator)
    for (; $i -lt $env_content.Length; $i++) {
        if ($env_content[$i][0] -eq '[') {
            $i++;
            break;
        }
        $p = $env_content[$i]
        $found = 0
        foreach ($v in $path_list) {
            if ($v -eq $p) {
                $found = 1
                break
            }
        }
        if ($found -eq 0) {
            $path_val = $p + [IO.Path]::PathSeparator + $path_val
        }
    }
    [System.Environment]::SetEnvironmentVariable("PATH", $path_val);
    # print path environment
    Write-Host $env:PATH
}

function Build-MMDeploy {
    if (-Not (Test-Path $envFile)) {
        Write-Warning "Can't find env.txt in $pwd, try to download dependencies"
        Write-Warning "please make sure you have edited the configure setting."
        Get-Dependences
        Write-EnvFile
    }

    Set-Environment

    $configureCommand = "cmake .. -A x64 -T v142"
    if ($MMDEPLOY_TARGET_BACKENDS.Contains("trt") -or $MMDEPLOY_TARGET_DEVICES.Contains("cuda")) {
        $configureCommand += ',cuda="' + $env:CUDA_PATH + '"'
    }
    $configureCommand += " -DMMDEPLOY_BUILD_SDK_CSHARP_API=${MMDEPLOY_BUILD_SDK_CSHARP_API}"
    $configureCommand += " -DMMDEPLOY_BUILD_SDK_PYTHON_API=${MMDEPLOY_BUILD_SDK_PYTHON_API}"
    $configureCommand += " -DMMDEPLOY_BUILD_SDK=${MMDEPLOY_BUILD_SDK}"
    $configureCommand += " -DMMDEPLOY_TARGET_DEVICES='${MMDEPLOY_TARGET_DEVICES}'"
    $configureCommand += " -DMMDEPLOY_TARGET_BACKENDS='${MMDEPLOY_TARGET_BACKENDS}'"
    $configureCommand += " -DMMDEPLOY_CODEBASES='${MMDEPLOY_CODEBASES}'"
    $configureCommand += " -DMMDEPLOY_BUILD_EXAMPLES=${MMDEPLOY_BUILD_EXAMPLES}"
    foreach ($key in $EnvVars.Keys) {
        $configureCommand += " -D${key}=$($EnvVars.$key)"
    }

    $buildDir = [IO.Path]::Combine($pwd, "build")
    if (Test-Path $buildDir -PathType Container) {
        Remove-Item $buildDir -Recurse
    }

    New-Item -ItemType "directory" -Path $buildDir
    $oldWorkingDir = $pwd
    Set-Location -Path $buildDir

    Write-Host $configureCommand
    Invoke-Expression $configureCommand

    Invoke-Expression "cmake --build . --config $CMAKE_BUILD_TYPE -j"
    Invoke-Expression "cmake --install . --config $CMAKE_BUILD_TYPE"
    Set-Location -Path $oldWorkingDir
}

function Install-Dependences {
    # [environment path]
    # [other path]
    param ()

    if (-Not (Test-Path $Dir -PathType Container)) {
        Write-Error "Folder $Dir doesn't exits"
    }

    if (-Not (Test-Path $envFile)) {
        Write-Error "Can't find env.txt in $pwd, Please first download depencencies"
    }
    $env_content = Get-Content $envFile
    Write-Host "Install Dependences to executable folder"

    $i = 1
    for (; $i -lt $env_content.Length; $i++) {
        if ($env_content[$i].StartsWith("[")) {
            $i++
            break
        }
    }
    # environment path
    for (; $i -lt $env_content.Length; $i++) {
        if ($env_content[$i].StartsWith("[")) {
            $i++;
            break;
        }
        $visit = $env_content[$i]
        Write-Host "checking folder: $visit"
        $dlls = (Get-ChildItem $env_content[$i] -Filter "*.dll").FullName
        foreach ($dll in $dlls) {
            Write-Host "Copy $dll to $Dir"
            Copy-Item -Path $dll -Destination $Dir
        }
    }
}

if ($Action -eq "Build") {
    Build-MMDeploy
}
elseif ($Action -eq "Download") {
    Get-Dependences
    Write-EnvFile
}
elseif ($Action -eq "Install") {
    Install-Dependences
}
elseif ($Action -eq "SetupEnv") {
    Set-Environment
}
else {
    Show-HelpMessage
}
