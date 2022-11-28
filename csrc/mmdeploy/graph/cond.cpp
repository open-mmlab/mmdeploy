// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/graph/cond.h"

#include <algorithm>

namespace mmdeploy::graph {

namespace {

std::vector<int> get_predicates(const Value::Array& xs) {
  std::vector<int> ps;
  ps.reserve(xs.size());
  std::transform(std::begin(xs), std::end(xs), std::back_inserter(ps),
                 [](const Value& x) { return static_cast<int>(x.get<bool>()); });
  return ps;
}

std::pair<bool, int> choice(const std::vector<int>& xs) {
  auto count = std::count(std::begin(xs), std::end(xs), 1);
  if (count == 0 || count == xs.size()) {
    return std::make_pair(true, count == xs.size());
  }
  return std::make_pair(false, false);
}

Value get_divergent_input(Value::Array& as, const std::vector<int>& ps) {
  Value::Array ts(as.size(), Value::kArray);
  for (size_t i = 0; i < ts.size(); ++i) {
    auto& t = ts[i].array();
    auto& a = as[i].array();
    for (size_t j = 0; j < ps.size(); ++j) {
      if (ps[j]) {
        t.push_back(std::move(a[j]));
      }
    }
  }
  return ts;
}

Value get_divergent_output(Value::Array& rs, const vector<int>& ps) {
  Value::Array ys(rs.size(), Value::kArray);
  for (size_t i = 0; i < ys.size(); ++i) {
    auto& y = ys[i].array();
    auto& r = rs[i].array();
    size_t k = 0;
    for (const auto& p : ps) {
      y.push_back(p ? std::move(r[k++]) : nullptr);
    }
  }
  return ys;
}

}  // namespace

Sender<Value> Cond::Process(Sender<Value> input) {
  auto index = std::make_shared<profiler::Index>();
  if (scope_) {
    *index = scope_->next_.fetch_add(1, std::memory_order_relaxed);
    input = Then(std::move(input), [this, index](Value v) mutable {
      scope_->Add(profiler::Event::kStart, *index, profiler::Clock::now());
      return std::move(v);
    });
  }

  Sender<Value> output = LetValue(std::move(input), [this](Value& _input) -> Sender<Value> {
    assert(_input.is_array());
    auto& as = _input.array();
    auto ps = get_predicates(as.front().array());
    as.erase(as.begin());
    auto [coherent, branch] = choice(ps);
    if (coherent) {
      if (branch) {
        return node_->Process(Just(std::move(_input)));
      } else {
        Value::Array output(n_output_, Value::Array(ps.size(), nullptr));
        return Just(Value(std::move(output)));
      }
    } else {
      auto ts = get_divergent_input(as, ps);
      return node_->Process(Just(Value(std::move(ts)))) |
             Then([ps = std::move(ps)](Value rs) -> Value {
               return get_divergent_output(rs.array(), ps);
             });
    }
  });

  if (scope_) {
    output = Then(std::move(output), [this, index](Value v) {
      scope_->Add(profiler::Event::kEnd, *index, profiler::Clock::now());
      return std::move(v);
    });
  }
  return output;
}

CondBuilder::CondBuilder(Value config) : Builder(std::move(config)) {}

Result<unique_ptr<Node>> CondBuilder::BuildImpl() {
  try {
    auto cond = std::make_unique<Cond>();
    cond->n_output_ = static_cast<int>(config_["output"].size());

    auto& body_config = config_["body"];
    auto inputs = config_["input"].array();
    inputs.erase(inputs.begin());

    body_config["input"] = std::move(inputs);
    body_config["output"] = config_["output"];

    // propagate context
    if (!body_config.contains("context")) {
      body_config["context"] = Value::Object();
    }
    if (config_.contains("context")) {
      update(body_config["context"].object(), config_["context"].object(), 2);
      if (config_["context"].contains("scope")) {
        auto scope = config_["context"]["scope"].get<profiler::Scope*>();
        auto name = config_.value("name", std::string("Cond"));
        cond->scope_ = scope->CreateScope(name);
        body_config["context"]["scope"] = cond->scope_;
      }
    }

    if (auto builder = Builder::CreateFromConfig(body_config).value()) {
      if (auto node = builder->Build().value()) {
        cond->node_ = std::move(node);
        return std::move(cond);
      }
    }
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("error parsing config: {}", config_);
  }
  return Status(eFail);
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Builder, (Cond, 0), [](const Value& config) {
  return std::make_unique<CondBuilder>(config);
})

}  // namespace mmdeploy::graph
