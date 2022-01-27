import os
import requests


def gen_cmd(global_json: dict,
            codebase_json: dict,
            mode: str,
            codebase: str,
            task: str,
            backend: str,
            model_id: int,
            deploy_cfg_id: int,
            resolution: str = None,
            run=False):
    model_config = codebase_json["models"][task][model_id]
    deploy_root = os.path.join(global_json['root'], 'mmdeploy')
    codebase_name = global_json['codebases'][codebase]['full_name']
    codebase_root = os.path.join(global_json['root'], codebase_name)
    task_config = codebase_json['deploy_cfgs'][task]
    backend_config = global_json['backends'][backend]
    task_foloder = task_config['folder']
    if resolution is None:
        if 'resolution' in model_config:
            resolution = model_config['resolution']
        else:
            resolution = task_config['default']

    metric = task_config['metric']
    if len(metric) > 0:
        metric = f"--metrics {metric}"

    deploy_config_path = task_config['cfgs'][resolution][backend][deploy_cfg_id]
    final_deploy_config_path = os.path.join(
        deploy_root, "configs", codebase, task_foloder, deploy_config_path)

    model_config_prefix = model_config["prefix"]
    model_config_path = model_config["cfg"]
    model_ckpt_path = model_config["ckpt"]
    final_model_config = os.path.join(
        codebase_root, "configs", model_config_prefix, model_config_path)
    final_ckpt = os.path.join(
        codebase_root, "checkpoints", model_ckpt_path)

    if not os.path.exists(final_ckpt):
        os.makedirs(os.path.join(
            codebase_root, "checkpoints", model_config_prefix), exist_ok=True)
        url = 'https://download.openmmlab.com/{codebase_name}/{model_config_prefix}/{model_ckpt_path}'
        file = requests.get(url)
        open(final_ckpt, 'wb').write(file.content)

    export_img = os.path.join(codebase_root, task_config['export_img'])
    test_img = os.path.join(codebase_root, task_config['test_img'])

    model_name = model_config_prefix.split('/')[-1]

    backend_name = backend_config['full_name']
    precision = backend_config['precision'][deploy_cfg_id]
    extra_info = f'{resolution}-{precision}'
    work_dir = os.path.join(
        'work_dirs', "output", codebase, model_name, backend_name, extra_info
    )
    device = backend_config['device']

    output_file = os.path.join(work_dir, 'output.txt')
    backend_files = ' '
    for ext in backend_config['backend_files']:
        backend_files += os.path.join(work_dir, f'end2end{ext}')

    if mode == 'deploy':
        command_str = f"python -W ignore\
            {deploy_root}/tools/deploy.py \
            {final_deploy_config_path} \
            {final_model_config} \
            {final_ckpt}\
            {export_img}\
            --test-img {test_img}\
            --work-dir {work_dir} \
            --device {device} \
            --log-level INFO \
            "
    elif mode == 'test':

        command_str = f"python\
        {deploy_root}/tools/test.py \
        {final_deploy_config_path} \
        {final_model_config} \
        --model {backend_files}\
        {metric} \
        --device {device} \
        --log2file {output_file} \
        "

    if run:
        print()
        print(50 * '----')
        print(command_str)
        if os.system(command_str) != 0:
            os.makedirs('work_dirs', exist_ok=True)
            with open('work_dirs/test_log.txt', 'a') as fp:
                fp.write(
                    f"Error when {mode} {model_name}-{backend_name}-{extra_info}\n")

    return command_str
