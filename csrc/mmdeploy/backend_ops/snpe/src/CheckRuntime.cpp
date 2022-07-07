//==============================================================================
//
//  Copyright (c) 2017-2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>

#include "CheckRuntime.hpp"

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/DlVersion.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/String.hpp"

// Command line settings
zdl::DlSystem::Runtime_t checkRuntime(zdl::DlSystem::Runtime_t runtime, bool &staticQuantization)
{
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();

    std::cout << "SNPE Version: " << Version.asString().c_str() << std::endl; //Print Version number

   if((runtime != zdl::DlSystem::Runtime_t::DSP) && staticQuantization)
   {
      std::cerr << "ERROR: Cannot use static quantization with CPU/GPU runtimes. It is only designed for DSP/AIP runtimes.\n";
      std::cerr << "ERROR: Proceeding without static quantization on selected runtime.\n";
      staticQuantization = false;
   }

    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime))
    {
        std::cerr << "Selected runtime not present. Falling back to CPU." << std::endl;
        runtime = zdl::DlSystem::Runtime_t::CPU;
    }

    return runtime;
}
bool checkGLCLInteropSupport()
{
    return zdl::SNPE::SNPEFactory::isGLCLInteropSupported();
}
