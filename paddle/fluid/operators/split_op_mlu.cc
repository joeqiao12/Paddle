/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>

#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/operators/split_op.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class SplitMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // init
    auto* in = ctx.Input<framework::Tensor>("X");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    int num = ctx.Attr<int>("num");
    std::vector<int> sections = ctx.Attr<std::vector<int>>("sections");
    int axis = ctx.Attr<int>("axis");
    auto in_dims = in->dims();
    auto outnumbers = outs.size();
    VLOG(2) << "in_dims------------------------------" << in_dims;
    VLOG(2) << "outnumbers-----------------------------" << outnumbers;

    if (ctx.HasInput("AxisTensor")) {
      PADDLE_THROW(platform::errors::Unimplemented(
          "The AxisTensor is not supported on MLU now."));
    }
    if (ctx.HasInput("SectionsTensorList")) {
      PADDLE_THROW(platform::errors::Unimplemented(
          "The SectionsTensorList is not supported on MLU now."));
    }

    // init out tensors
    std::vector<void*> vct_tensor;
    // std::vector<const Tensor*> vct_tensor;
    std::vector<MLUCnnlTensorDesc> output_descs;
    std::vector<cnnlTensorDescriptor_t> desc_vector;
    auto place = ctx.GetPlace();
    for (size_t i = 0; i < outs.size(); i++) {
      outs[i]->mutable_data<T>(ctx.GetPlace());
      output_descs.emplace_back(MLUCnnlTensorDesc(
          *outs[i], CNNL_LAYOUT_ARRAY, ToCnnlDataType(outs[i]->type())));
      desc_vector.push_back(output_descs.back().get());
      vct_tensor.push_back(reinterpret_cast<void*>(outs[i]->data<T>()));
      // vct_tensor.push_back(reinterpret_cast<const
      // Tensor*>(outs[i]->data<T>()));
    }
    // init in tensors
    MLUCnnlTensorDesc input_desc(*in, CNNL_LAYOUT_ARRAY,
                                 ToCnnlDataType(in->type()));

    // MLU should do sth
    MLUCnnl::Split(ctx, num, axis, input_desc.get(),
                   reinterpret_cast<const void*>(in->data<T>()),
                   desc_vector.data(), vct_tensor.data());
    // MLUCnnl::Split(ctx, num, axis,
    //                input_desc.get(), reinterpret_cast<const
    //                void*>(in->data<T>()),
    //                reinterpret_cast<const
    //                CnnlTensorDesc*>(desc_vector.data()), vct_tensor.data());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(split, ops::SplitMLUKernel<float>,
                       ops::SplitMLUKernel<int>,
                       ops::SplitMLUKernel<plat::float16>);
