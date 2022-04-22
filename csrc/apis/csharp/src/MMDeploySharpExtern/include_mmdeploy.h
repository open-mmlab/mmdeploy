#pragma once

#ifndef CV_EXPORTS
#if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#define CV_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4 && defined(__APPLE__)
#define CV_EXPORTS __attribute__((visibility("default")))
#else
#define CV_EXPORTS
#endif
#endif

#if defined WIN32 || defined _WIN32
#define CV_CDECL __cdecl
#define CV_STDCALL __stdcall
#else
#define CV_CDECL
#define CV_STDCALL
#endif

#ifndef MMDEPLOYAPI
#define MMDEPLOYAPI(rettype) extern "C" CV_EXPORTS rettype CV_CDECL
#endif

#include "classifier.h"
#include "common.h"
#include "detector.h"
#include "model.h"
#include "pose_detector.h"
#include "restorer.h"
#include "segmentor.h"
#include "text_detector.h"
#include "text_recognizer.h"