//==============================================================================
//
//  Copyright (c) 2017-2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef CHECKRUNTIME_H
#define CHECKRUNTIME_H

#include "SNPE/SNPEFactory.hpp"

zdl::DlSystem::Runtime_t checkRuntime(zdl::DlSystem::Runtime_t runtime, bool &staticQuantization);
bool checkGLCLInteropSupport();

#endif
