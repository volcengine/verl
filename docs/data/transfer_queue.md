# TransferQueue Data System

Last updated: 09/28/2025.

This doc introduce [TransferQueue](https://github.com/TransferQueue/TransferQueue), an asynchronous streaming data management system for efficient post-training.


<h2 id="overview"> Overview</h2>

TransferQueue is a high-performance data storage and transfer system with panoramic data visibility and streaming scheduling capabilities, optimized for efficient dataflow in post-training workflows.

<p align="center">
  <img src="https://cdn.nlark.com/yuque/0/2025/png/23208217/1758696193102-a5654375-65a1-4e06-9c63-142b59df90b8.png" width="70%">
</p>


TransferQueue offers **fine-grained, sample-level** data management capabilities, serving as a data gateway that decouples explicit data dependencies across computational tasks. This enables a divide-and-conquer approach, significantly simplifying the design of the algorithm controller.


<p align="center">
  <img src="https://cdn.nlark.com/yuque/0/2025/png/23208217/1758696791245-fa7baf96-46af-4c19-8606-28ffadc4556c.png" width="70%">
</p>




<h2 id="components"> Components</h2>



### Control Plane: Panoramic Data Management  

In the control plane, `TransferQueueController` tracks the **production status** and **consumption status** of each training sample as metadata. When all the required data fields are ready (i.e., written to the `TransferQueueStorage`), we know that this data sample can be consumed by downstream tasks. 

For consumption status, we record the consumption records for each computational task (e.g., `generate_sequences`, `compute_log_prob`, etc.). Therefore, even different computation tasks require the same data field, they can consume the data independently without interfering with each other.


<p align="center">
  <img src="https://cdn.nlark.com/yuque/0/2025/png/23208217/1758696820173-456c1784-42ba-40c8-a292-2ff1401f49c5.png" width="70%">
</p>


> In the future, we plan to support **load-balancing** and **dynamic batching** capabilities in the control plane. Besides, we will support data management for disaggregated frameworks where each rank manages the data retrieval by itself, rather than coordinated by a single controller.

### Data Plane: Distributed Data Storage

In the data plane, `TransferQueueStorageSimpleUnit` serves as a naive storage unit based on CPU memory, responsible for the actual storage and retrieval of data. Each storage unit can be deployed on a separate node, allowing for distributed data management.

`TransferQueueStorageSimpleUnit` employs a 2D data structure as follows:

- Each row corresponds to a training sample, assigned a unique index within the corresponding global batch.
- Each column represents the input/output data fields for computational tasks.

This data structure design is motivated by the computational characteristics of the post-training process, where each training sample is generated in a relayed manner across task pipelines. It provides an accurate addressing capability, which allows fine-grained, concurrent data read/write operations in a streaming manner.

<p align="center">
  <img src="https://cdn.nlark.com/yuque/0/2025/png/23208217/1758696805154-3817011f-84e6-40d0-a80c-58b7e3e5f6a7.png" width="70%">
</p>


> In the future, we plan to implement a **general storage abstraction layer** to support various storage backends. Through this abstraction, we hope to integrate high-performance storage solutions such as [MoonCakeStore](https://github.com/kvcache-ai/Mooncake) to support device-to-device data transfer through RDMA, further enhancing data transfer efficiency for large-scale data.


### User Interface: Asynchronous & Synchronous Client


The interaction workflow of TransferQueue system is as follows:

1. A process sends a read request to the `TransferQueueController`.
2. `TransferQueueController` scans the production and consumption metadata for each sample (row), and dynamically assembles a micro-batch metadata according to the load-balancing policy. This mechanism enables sample-level data scheduling.
3. The process retrieves the actual data from distributed storage units using the metadata provided by the controller.

To simplify the usage of TransferQueue, we have encapsulated this process into `AsyncTransferQueueClient` and `TransferQueueClient`. These clients provide both asynchronous and synchronous interfaces for data transfer, allowing users to easily integrate TransferQueue to their framework.


> In the future, we will provide a `StreamingDataLoader` interface for disaggregated frameworks as discussed in [RFC#2662](https://github.com/volcengine/verl/discussions/2662). Leveraging this abstraction, each rank can automatically get its own data like `DataLoader` in PyTorch. The TransferQueue system will handle the underlying data scheduling and transfer logic caused by different parallelism strategies, significantly simplifying the design of disaggregated frameworks.


<h2 id="show-cases"> Show Cases</h2>

### General Usage

The primary interaction points are `AsyncTransferQueueClient` and `TransferQueueClient`, serving as the communication interface with the TransferQueue system.

Core interfaces:

- (async_)get_meta(data_fields: list[str], batch_size:int, global_step:int, get_n_samples:bool, task_name:str) -> BatchMeta
- (async_)get_data(metadata:BatchMeta) -> TensorDict
- (async_)put(data:TensorDict, metadata:BatchMeta, global_step)
- (async_)clear(global_step: int)


We will soon release a detailed tutorial and API documentation.


### verl Example


The primary motivation for integrating TransferQueue to verl now is to **alleviate the data transfer bottleneck of the single controller `RayPPOTrainer`**. Currently,  all `DataProto` objects must be routed through `RayPPOTrainer`, resulting in a single point bottleneck of the whole post-training system. 

![verl_dataflow_DataProto](https://cdn.nlark.com/yuque/0/2025/jpeg/23208217/1758704289414-bcc54228-716b-4d4a-ad3b-f9ace6d10fcf.jpeg)

Leveraging TransferQueue, we separate experience data transfer from metadata dispatch by

- Replacing `DataProto` with `BatchMeta` (metadata) and `TensorDict` (actual data) structures
- Preserving verl's original Dispatch/Collect logic via BatchMeta (maintaining single-controller debuggability)
- Accelerating data transfer by TransferQueue's distributed storage units

![verl_dataflow_TransferQueue](https://cdn.nlark.com/yuque/0/2025/jpeg/23208217/1758704301666-0807dc06-766c-4a2d-9cde-889a6bb56b34.jpeg)


You may refer to the [recipe](https://github.com/TransferQueue/TransferQueue/tree/dev/recipe/simple_use_case), where we mimic the verl usage in both async & sync scenarios. 





<h2 id="citation"> Citation</h2>
Please kindly cite our paper if you find this repo is useful:

```bibtex
@article{han2025asyncflow,
  title={AsyncFlow: An Asynchronous Streaming RL Framework for Efficient LLM Post-Training},
  author={Han, Zhenyu and You, Ansheng and Wang, Haibo and Luo, Kui and Yang, Guang and Shi, Wenqi and Chen, Menglong and Zhang, Sicheng and Lan, Zeshun and Deng, Chunshi and others},
  journal={arXiv preprint arXiv:2507.01663},
  year={2025}
}
```