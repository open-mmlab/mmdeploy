/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Copyright (c) OpenMMLab. All rights reserved.

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

#include <iostream>

#include "service_impl.h"
#include "text_table.h"

void PrintIP() {
  struct ifaddrs* ifAddrStruct = NULL;
  void* tmpAddrPtr = NULL;

  int retval = getifaddrs(&ifAddrStruct);
  if (retval == -1) {
    return;
  }

  helper::TextTable table("Device");
  table.padding(1);
  table.add("port").add("ip").eor();
  while (ifAddrStruct != nullptr) {
    if (ifAddrStruct->ifa_addr == nullptr) {
      break;
    }

    if (ifAddrStruct->ifa_addr->sa_family == AF_INET) {
      tmpAddrPtr = &((struct sockaddr_in*)ifAddrStruct->ifa_addr)->sin_addr;
      char addressBuffer[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, tmpAddrPtr, addressBuffer, INET_ADDRSTRLEN);
      table.add(std::string(ifAddrStruct->ifa_name)).add(std::string(addressBuffer)).eor();
    } else if (ifAddrStruct->ifa_addr->sa_family == AF_INET6) {
      tmpAddrPtr = &((struct sockaddr_in*)ifAddrStruct->ifa_addr)->sin_addr;
      char addressBuffer[INET6_ADDRSTRLEN];
      inet_ntop(AF_INET6, tmpAddrPtr, addressBuffer, INET6_ADDRSTRLEN);
      table.add(std::string(ifAddrStruct->ifa_name)).add(std::string(addressBuffer)).eor();
    }
    ifAddrStruct = ifAddrStruct->ifa_next;
  }
  std::cout << table << std::endl << std::endl;
}

void RunServer(int port = 60000) {
  // listen IPv4 and IPv6
  char server_address[64] = {0};
  sprintf(server_address, "[::]:%d", port);
  InferenceServiceImpl service;

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

  // Max 128MB
  builder.SetMaxMessageSize(2 << 29);
  builder.SetMaxSendMessageSize(2 << 29);

  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.

  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<Server> server(builder.BuildAndStart());
  fprintf(stdout, "Server listening on %s\n", server_address);

  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  server->Wait();
}

int main(int argc, char** argv) {
  int port = 60000;
  if (argc > 1) {
    port = std::stoi(argv[1]);
  }

  if (port <= 9999) {
    fprintf(stdout, "Usage: %s [port]\n", argv[0]);
    return 0;
  }
  PrintIP();
  RunServer(port);

  return 0;
}
