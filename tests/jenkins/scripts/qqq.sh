## parameters
config="../conf/${1:-default.config}"
docker_image=$(grep docker_image ${config} | sed 's/docker_image=//')
codebase_list=($(grep codebase_list ${config} | sed 's/codebase_list=//'))
exec_performance=$(grep exec_performance ${config} | sed 's/exec_performance=//')
max_job_nums=$(grep max_job_nums ${config} | sed 's/max_job_nums=//')
mmdeploy_branch=$(grep mmdeploy_branch ${config} | sed 's/mmdeploy_branch=//')
repo_url=$(grep repo_url ${config} | sed 's/repo_url=//')

echo ${config}
echo ${docker_image}
echo ${codebase_list}
for i in {1..2}
do
    echo $codebase_list[i]
done