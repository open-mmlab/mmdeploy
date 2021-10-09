## How to convert model

```bash
python ./tools/deploy.py \
    ${DEPLOY_CFG_PATH} \
    ${MODEL_CFG_PATH} \
    ${MODEL_CHECKPOINT_PATH} \
    ${INPUT_IMG} \
    --work-dir ${WORK_DIR} \
    --device ${DEVICE} \
    --log-level INFO
```
