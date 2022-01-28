import os
import json

from impl import gen_cmd


root = 'auto_test'
test_codebases = [
    "mmcls",
    "mmedit",
    "mmseg",
    "mmocr",
    "mmdet",
    "mmpose"
]


def main():
    with open(os.path.join(root, 'cfg', 'global.json'), "r") as fp:
        global_cfg = json.load(fp)
    assert os.path.exists(global_cfg['root']), f'File not exists: {global_cfg["root"]}'
    for codebase in test_codebases:
        with open(os.path.join(root, 'cfg', f'{codebase}.json'), "r") as fp:
            codebase_cfg = json.load(fp)
        for task in codebase_cfg['models']:
            for model_id in range(len(codebase_cfg['models'][task])):
                for backend, backend_dict in global_cfg['backends'].items():
                    for deploy_cfg_id in range(len(backend_dict['precision'])):
                        try:
                            gen_cmd(global_cfg, codebase_cfg, 'deploy', codebase,
                                    task, backend, model_id,
                                    deploy_cfg_id, run=True)
                            if backend_dict['do_test']:
                                gen_cmd(global_cfg, codebase_cfg, 'test', codebase,
                                        task, backend, model_id,
                                        deploy_cfg_id, run=True)
                        except Exception:
                            os.makedirs('work_dirs', exist_ok=True)
                            with open('work_dirs/test_log.txt', 'a') as fp:
                                fp.write(
                                    f"Config not exist: {codebase}-{model_id} {backend}-{deploy_cfg_id}")


if __name__ == '__main__':
    main()
