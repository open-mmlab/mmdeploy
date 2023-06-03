function ReadConfig() {
    param (
        [string] $config_path

    )
    Write-Host "read config path: $config_path"
    $conf = @{}
    Write-Host "---------------------- start reading config file ----------------------"
    $payload = Get-Content -Path $config_path |
    Where-Object { $_ -like '*=*' } |
    ForEach-Object {
        $infos = $_ -split '='
        $key = $infos[0].Trim()
        $value = $infos[1].Trim()
        $conf.$key = $value
    }
    Write-Host "---------------------- end reading config file ----------------------"
    return $conf
}
function InitMim() {
    param (
        [string] $codebase,
        [string] $codebase_fullname,
        [string] $mmdeploy_branch
    )
    $url = "https://github.com/open-mmlab/"+$codebase_fullname+".git"
    $place = (Join-Path $env:JENKINS_WORKSPACE $codebase_fullname)
#     $codebase_fullname = “mmdetection” | “mmclassification” | "mmaction2" | "mmpose" | "mmsegmentation" | "mmediting" | “mmocr” | “mmrotate” | “mmdetection3d” | “mmyolo”
    Write-Host "---------------------- start cloning $codebase_fullname ----------------------"
    Write-Host "mmdeploy_branch = $mmdeploy_branch "

    if ($mmdeploy_branch -eq "master" ){
        if ($codebase_fullname -eq “mmdetection” ){
            git clone --depth 1 -b v2.20.0 $url $place
        }
        elseif ($codebase_fullname -eq “mmrotate”  ){
            git clone --depth 1 -b main $url $place
        }
        elseif ($codebase_fullname -eq “mmclassification” ){
            git clone --depth 1 -b v1.0.0rc6 $url $place
        }
        elseif ($codebase_fullname -eq “mmdetection3d” ){
            git clone --depth 1 -b v1.0.0rc4 $url $place
        }
        elseif ($codebase_fullname -eq “mmocr” -or $codebase_fullname -eq "mmpose" -or $codebase_fullname -eq "mmaction2" `
            -or $codebase_fullname -eq "mmsegmentation" -or $codebase_fullname -eq "mmediting"){
            git clone --depth 1 -b 0.x $url $place
        }

    }
    elseif ($mmdeploy_branch -eq "main") {
        if ($codebase_fullname -eq “mmocr” -or $codebase_fullname -eq “mmdetection” `
        -or $codebase_fullname -eq "mmpose"-or $codebase_fullname -eq "mmsegmentation" -or $codebase_fullname -eq "mmyolo" `
        -or $codebase_fullname -eq "mmediting"-or $codebase_fullname -eq "mmaction2" -or $codebase_fullname -eq "mmdetection3d"){
            git clone --depth 1 -b main $url $place
        }
        elseif ($codebase_fullname -eq “mmclassification”){
            git clone --depth 1 -b mmcls-1.x $url $place
        }
        elseif ($codebase_fullname -eq “mmrotate” ){
            git clone --depth 1 -b dev-1.x $url $place
        }
    }

    Write-Host "---------------------- end cloning $codebase_fullname ----------------------"
}

$NV_EXT = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{0}\extras\visual_studio_integration\MSBuildExtensions\"
$MS_EXT="C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations"

function SwitchCudaVersion() {
    param (
        [string] $cuda_version
    )
    if ($cuda_version -eq "cu111") {
        $cuda = "11.1"
        Write-Host "switch cuda version to cu111"
        $env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1"
        $env:CUDA_PATH_V11_3=$null
        $env:CUDA_PATH_V11_1="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1"
        $env:path="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\libnvvp;"+$env:path
        $old = (Join-PATH $MS_EXT *)
        Remove-Item $old -Include "CUDA *.props", "CUDA *.xml", "CUDA *.targets", "Nvda.Build.CudaTasks*.dll"
        $new = (Join-PATH ($NV_EXT -f $cuda) *)
        Copy-Item -Path $new -Destination $MS_EXT
    } elseif ($cuda_version -eq "cu113") {
        $cuda = "11.3"
        Write-Host "switch cuda version to cu113"
        $env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"
        $env:CUDA_PATH_V11_1=$null
        $env:CUDA_PATH_V11_3="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3"
        $env:path="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\libnvvp;"+$env:path
        $old = (Join-PATH $MS_EXT *)
        #Remove-Item $old -Include "CUDA *.props", "CUDA *.xml", "CUDA *.targets", "Nvda.Build.CudaTasks*.dll"
        $new = (Join-PATH ($NV_EXT -f $cuda) *)
        #Copy-Item -Path $new -Destination $MS_EXT
    }
}
