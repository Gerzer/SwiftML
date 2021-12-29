//
//  Graph.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 3/8/21.
//

import MLCompute

/// Represents an executable graph of layers.
public class Graph {
	
	/// Builds a graph of layers.
	/// - Warning: Don’t use this structure directly.
	@resultBuilder public struct Builder {
		
		public static func buildBlock(_ layers: Layer...) -> [Layer] {
			return layers
		}
		
	}
	
	private var layers: [Layer]
	
	private var weightsContainer: WeightsContainer?
	
	public init(@Builder _ builder: () -> [Layer]) {
		self.layers = builder()
	}
	
	/// Executes the graph for the specified number of training iterations.
	/// - Parameters:
	///   - data: The input and target tensors.
	///   - iterations: The number of iterations for which to train.
	///   - device: The device on which to execute the graph.
	///   - callback: A callback that’s executed for each output tensor from the final training iteration.
	/// - Throws: ``TrainingError/incompatible``, ``TrainingError/alreadyTrained``, ``DeviceError/noGPUDetected``
	public func train(data: TrainingData, iterations: Int, on device: TrainingComputeDevice, callback: @escaping (Tensor) -> Void) throws {
		for layer in self.layers {
			if !layer.checkTrainingCompatibility(on: device) {
				throw TrainingError.incompatible
			}
		}
		precondition(iterations > 0, "Invalid iterations count")
		precondition(data.inputTensors.count > 0, "Must supply at least one input tensor")
		precondition(data.inputTensors.count == data.targetTensors.count, "Must supply the same number of input tensors and target tensors")
		let inputSameShape = data.inputTensors
			.dropFirst()
			.allSatisfy { (tensor) in
				return tensor.shape == data.inputTensors.first!.shape
			}
		precondition(inputSameShape, "All input tensors must have the same shape")
		let targetSameShape = data.targetTensors
			.dropFirst()
			.allSatisfy { (tensor) in
				return tensor.shape == data.targetTensors.first!.shape
			}
		precondition(targetSameShape, "All target tensors must have the same shape")
		precondition(data.inputTensors.first!.shape.batchSize == data.targetTensors.first!.shape.batchSize, "Input tensors and target tensors must have the same batch size")
		guard self.weightsContainer == nil else {
			throw TrainingError.alreadyTrained
		}
		let internalDevice = try device.select()
		let internalGraph = MLCGraph()
		var internalTensors = [MLCTensor(shape: data.inputTensors.first!.shape.shapeArray)]
		var nextInputShape = data.inputTensors.first!.shape
		for layerIndex in self.layers.indices {
			self.layers[layerIndex].configure(inputShape: nextInputShape, on: internalDevice)
			nextInputShape = self.layers[layerIndex].outputShape
			let tensor = internalGraph.node(with: self.layers[layerIndex].internalLayer, sources: [internalTensors.last!])!
			print(tensor.descriptor.shape)
			internalTensors.append(tensor)
		}
		precondition(internalTensors.last!.descriptor.shape == data.targetTensors.first!.shape.shapeArray, "Target shape mismatch")
		let trainingGraph = MLCTrainingGraph(
			graphObjects: [internalGraph],
			lossLayer: MLCLossLayer.meanSquaredError(
				reductionType: .none,
				weight: 1
			),
			optimizer: MLCAdamOptimizer(
				descriptor: MLCOptimizerDescriptor(
					learningRate: 0.01,
					gradientRescale: 1,
					regularizationType: .none,
					regularizationScale: 1
				)
			)
		)
		for internalLayer in trainingGraph.layers {
			internalLayer.isDebuggingEnabled = true
		}
		let internalInputTensor = internalTensors.first!
		let internalOutputTensor = internalTensors.last!
		let internalTargetTensor = MLCTensor(shape: data.targetTensors.first!.shape.shapeArray)
		let internalLossResultTensors = trainingGraph.resultTensors(for: trainingGraph.layers.last!)
		let inputs = [
			internalInputTensor.label: internalInputTensor
		]
		let targets = [
			internalTargetTensor.label: internalTargetTensor
		]
		trainingGraph.addInputs(inputs, lossLabels: targets)
		trainingGraph.compile(options: [.debugLayers], device: internalDevice)
		for iteration in 0 ..< iterations {
			print("Executing iteration \(iteration + 1) of \(iterations)...")
			for (inputTensor, targetTensor) in zip(data.inputTensors, data.targetTensors) {
				let inputTensorData = inputTensor.internalTensorData()
				let targetTensorData = targetTensor.internalTensorData()
				let inputsData = [
					internalInputTensor.label: inputTensorData
				]
				let targetsData = [
					internalTargetTensor.label: targetTensorData
				]
				trainingGraph.execute(inputsData: inputsData, lossLabelsData: targetsData, lossLabelWeightsData: nil, batchSize: inputTensor.shape.batchSize, options: [.synchronous]) { (_, _, _) in
					print("\nLoss result tensors:")
					for internalLossResultTensor in internalLossResultTensors {
						print(try! internalLossResultTensor.dataArray(as: Float.self))
					}
					if iteration == iterations - 1 {
						// TODO: Handle throwing initializer more gracefully
						callback(try! Tensor(from: internalOutputTensor))
					}
				}
			}
		}
		self.weightsContainer = WeightsContainer()
		for layer in self.layers {
			try layer.storeWeights(in: &self.weightsContainer!)
		}
	}
	
	/// Executes the graph for one inference iteration.
	/// - Parameters:
	///   - inputTensor: The input tensor to the first layer.
	///   - device: The device on which to execute the graph.
	///   - doIgnoreBatchSize: Whether to ignore all batch elements in the output tensor besides the first (*i.e.*, the batch element at index `0`).
	/// - Throws: ``InferenceError/incompatible``, ``DeviceError/noGPUDetected``, ``DeviceError/noANEDetected``
	/// - Returns: The output tensor from the last layer.
	public func infer(from inputTensor: Tensor, on device: InferenceComputeDevice, ignoreBatchSize doIgnoreBatchSize: Bool = true) throws -> Tensor {
		for layer in self.layers {
			if !layer.checkInferenceCompatibility(on: device) {
				throw InferenceError.incompatible
			}
		}
		if self.weightsContainer == nil {
			print("Warning: This graph has not yet been trained")
		}
		let internalDevice = try device.select()
		let internalGraph = MLCGraph()
		var internalTensors = [MLCTensor(shape: inputTensor.shape.shapeArray)]
		var nextInputShape = inputTensor.shape
		for layerIndex in self.layers.indices {
			if let weightsContainer = self.weightsContainer {
				try self.layers[layerIndex].loadWeights(from: weightsContainer, at: layerIndex, on: internalDevice)
			}
			self.layers[layerIndex].configure(inputShape: nextInputShape, on: internalDevice)
			nextInputShape = self.layers[layerIndex].outputShape
			let tensor = internalGraph.node(with: self.layers[layerIndex].internalLayer, sources: [internalTensors.last!])!
			internalTensors.append(tensor)
		}
		let inferenceGraph = MLCInferenceGraph(graphObjects: [internalGraph])
		for internalLayer in inferenceGraph.layers {
			internalLayer.isDebuggingEnabled = true
		}
		inferenceGraph.compile(options: [.debugLayers], device: internalDevice)
		let internalInputTensor = internalTensors.first!
		let internalOutputTensor = internalTensors.last!
		let internalInputTensorData = inputTensor.internalTensorData()
		let inputs = [
			internalInputTensor.label: internalInputTensor
		]
		let inputsData = [
			internalInputTensor.label: internalInputTensorData
		]
		inferenceGraph.addInputs(inputs)
		inferenceGraph.execute(inputsData: inputsData, batchSize: inputTensor.shape.batchSize, options: [.synchronous])
		let outputTensor = try Tensor(from: internalOutputTensor)
		return doIgnoreBatchSize ? outputTensor.batchElements[0] : outputTensor
	}
	
}
