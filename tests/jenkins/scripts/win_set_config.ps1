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