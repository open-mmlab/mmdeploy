#pragma once

extern "C" {
void fuse_func(void* host_data_in, const char* platform_name, const char* info, void* data_out);
}
