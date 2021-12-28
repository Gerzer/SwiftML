# ``SwiftML``

Easy machine learning with hardware acceleration.

## Overview

SwiftML is a framework that enables easy training of and inference on neural networks and other machine-learning models with CPU, GPU, and Apple Neural Engine (ANE) hardware acceleration. It presents a simple, idiomatic Swift API that makes the construction of complex computation graphs elegant and effortless. SwiftML is also extensible: you can create your own custom layers that interoperate seamlessly with the built-in layers.

The API is designed to hide much of the complexity that’s involved in parallel computation on homogenous hardware setups. This necessitates intricate interplay between the core components of the framework and the user-extensible layers. If you want to create your own custom layers, then ensure that you follow all of the relevant documentation exactly. In particular, never call any public APIs that warn in their documentation not to call them directly. Typically, you implement these APIs yourself in each of your custom layers and let SwiftML invoke them for you.

SwiftML operates primarily on three-dimensional tensors (*i.e.*, tensors of rank `3`) with its built-in ``Tensor`` structure. User-provided tensors that aren’t of rank `3` are currently unsupported.

## Topics

### Data

- ``Tensor``
- ``TrainingData``
- ``WeightsContainer``

### Computation

- ``Layer``
- ``Graph``
- ``ComputeDevice``

### Errors

- ``DeviceError``
- ``InferenceError``
- ``TrainingError``

### Internal Types

- ``InternalTensors``
