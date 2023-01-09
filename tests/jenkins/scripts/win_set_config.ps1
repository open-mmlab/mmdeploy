param($cblist, $eperformance, $rurl, $mbranch, $winconfig)
Write-Host "$cblist, $eperformance, $rurl, $mbranch , $winconfig"
Write-Host "$pwd"
Copy-Item -Path $pwd/tests/jenkins/conf/$winconfig -Destination $pwd/tests/jenkins/conf/win_tmp.config -Recurse -Force
(Get-content ./tests/jenkins/conf/win_tmp.config) -replace 'codebase_list=.*','codebase_list=$cblist' | Set-Content ./tests/jenkins/conf/win_tmp.config -Verbos
(Get-content ./tests/jenkins/conf/win_tmp.config) -replace 'exec_performance=.*','exec_performance=$eperformance' | Set-Content ./tests/jenkins/conf/win_tmp.config -Verbos
(Get-content ./tests/jenkins/conf/win_tmp.config) -replace 'repo_url=.*','repo_url=$rurl' | Set-Content ./tests/jenkins/conf/win_tmp.config -Verbos
(Get-content ./tests/jenkins/conf/win_tmp.config) -replace 'mmdeploy_branch=.*','mmdeploy_branch=$mbranch' | Set-Content ./tests/jenkins/conf/win_tmp.config -Verbos