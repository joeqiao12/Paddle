/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/concat_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"

namespace paddle {
namespace operators {

template <typename T>
class ConcatMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::LoDTensor>("X");
    auto* out = ctx.Output<framework::LoDTensor>("Out");
    PADDLE_ENFORCE_NOT_NULL(ins[0],
                            platform::errors::NotFound(
                                "The first input tensor is not initalized."));
    auto axis = ctx.Attr<int>("axis");
    auto num = ins.size();
    // compute negative number
    axis = ComputeAxis(static_cast<int64_t>(axis),
                       static_cast<int64_t>(ins[0]->dims().size()));

    auto place = ctx.GetPlace();
    out->mutable_data<T>(place);

    // mlu should do sth
    // init ins tensors
    std::vector<void*> vct_tensor;
    std::vector<MLUCnnlTensorDesc> input_descs;
    std::vector<cnnlTensorDescriptor_t> desc_vector;
    auto place = ctx.GetPlace();
    for (size_t i = 0; i < ins.size(); i++) {
      ins[i]->mutable_data<T>(ctx.GetPlace());
      input_descs.emplace_back(MLUCnnlTensorDesc(
          *ins[i], CNNL_LAYOUT_ARRAY, ToCnnlDataType(ins[i]->type())));
      desc_vector.push_back(input_descs.back().get());
      vct_tensor.push_back(reinterpret_cast<void*>(ins[i]->data<T>()));
    }
    // init out tensors
    MLUCnnlTensorDesc output_desc(*out, CNNL_LAYOUT_ARRAY,
                                  ToCnnlDataType(out->type()));

    // MLU should do sth
    MLUCnnl::Concat(ctx, num, axis, desc_vector.data(), vct_tensor.data(),
                    output_desc.get(),
                    reinterpret_cast<const void*>(out->data<T>()));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(concat, ops::ConcatMLUKernel<float>,
                       ops::ConcatMLUKernel<paddle::platform::float16>,
                       ops::ConcatMLUKernel<int64_t>,
                       ops::ConcatMLUKernel<bool>, ops::ConcatMLUKernel<int>,
                       ops::ConcatMLUKernel<uint8_t>);
