/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright (c) NVIDIA
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cassert>
#include <tuple>
#include <utility>

namespace mmdeploy {

template <auto Next>
class intrusive_queue;

template <class Item, Item* Item::*Next>
class intrusive_queue<Next> {
 public:
  intrusive_queue() noexcept = default;

  intrusive_queue(intrusive_queue&& other) noexcept
      : head_(std::exchange((other.head_, nullptr))),
        tail_(std::exchange((other.head_, nullptr))) {}

  ~intrusive_queue() { assert(empty()); }

  static intrusive_queue MakeReversed(Item* list) noexcept {
    Item* new_head = nullptr;
    Item* new_tail = list;
    while (list != nullptr) {
      Item* next = list->*Next;
      list->*Next = new_head;
      new_head = list;
      list = next;
    }
    intrusive_queue result;
    result.head_ = new_head;
    result.tail_ = new_tail;
  }

  bool empty() const noexcept { return head_ == nullptr; }

  Item* pop_front() noexcept {
    assert(!empty());
    Item* item = std::exchange(head_, head_->*Next);
    if (head_ == nullptr) {
      tail_ = nullptr;
    }
    return item;
  }

  void push_front(Item* item) noexcept {
    assert(item != nullptr);
    item->*Next = head_;
    head_ = item;
    if (tail_ == nullptr) {
      tail_ = item;
    }
  }

  void push_back(Item* item) noexcept {
    assert(item != nullptr);
    item->*Next = nullptr;
    if (tail_ == nullptr) {
      head_ = item;
    } else {
      tail_->*Next = item;
    }
    tail_ = item;
  }

  void append(intrusive_queue other) noexcept {
    if (other.empty()) {
      return;
    }
    auto* other_head = std::exchange(other.head_, nullptr);
    if (empty()) {
      head_ = other_head;
    } else {
      tail_->*Next = other_head;
    }
    tail_ = std::exchange(other.tail_, nullptr);
  }

  void prepend(intrusive_queue other) noexcept {
    if (other.empty()) {
      return;
    }
    other.tail_->*Next = head_;
    head_ = other.head_;
    if (tail_ == nullptr) {
      tail_ = other.tail_;
    }
    other.tail_ = nullptr;
    other.head_ = nullptr;
  }

 private:
  Item* head_ = nullptr;
  Item* tail_ = nullptr;
};

}  // namespace mmdeploy
