// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_APIS_C_MODEL_H_
#define MMDEPLOY_SRC_APIS_C_MODEL_H_

#include "common.h"

int mmdeploy_model_create_by_path(const char* path, mm_model_t* model);

int mmdeploy_model_create(const void* buffer, int size, mm_model_t* model);

void mmdeploy_model_destroy(mm_model_t* model);

#endif  // MMDEPLOY_SRC_APIS_C_MODEL_H_
