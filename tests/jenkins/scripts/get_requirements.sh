repo=$1
version=$2

function (repo, version) {
    if [ repo -eq mmdet ] then {
        if [ version -eq v1.0 ] then {
            branch=master
            mmcv="mmcv-full==1.6.1"
            codebase_version="mmdet"
        }
    }

    return (branch mmcv codebase_version)
}
