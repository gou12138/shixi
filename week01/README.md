预热：消除 GPU 初始化、核函数编译的开销，让 GPU 进入稳定工作状态


同步：解决 GPU 异步执行的问题，确保计时的是 “GPU 实际运算的时间”


正确性验证：先判断算子正确再去评测和优化性能



@triton.jit：GPU 执行的程序，要求张量在内存中连续

offsets：warp中元素的偏移

mask：防止越界

cuda_threat----warp----block----grid