//
//  LSTMLayer.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 3/8/21.
//

import MLCompute

/// A long-short-term-memory (LSTM) layer.
public struct LSTMLayer: Layer {
	
	public private(set) var internalLayer: MLCLayer! = nil
	
	public private(set) var inputShape: Tensor.Shape! = nil
	
	public private(set) var outputShape: Tensor.Shape! = nil
	
	private let hiddenSize: Int
	
	private let layerCount: Int
	
	private let returnsSequences: Bool
	
	private var inputWeights: [MLCTensor]! = nil
	
	private var hiddenWeights: [MLCTensor]! = nil
	
	private var biases: [MLCTensor]! = nil
	
	private lazy var inputWeightsDataArrays = (0 ..< 4 * self.layerCount).map { (_) in
		return (0 ..< self.inputShape.primaryAxis * self.hiddenSize).map { (_) in
			return Float.random(in: -1 ... 1)
		}
	}
	
	private lazy var hiddenWeightsDataArrays = (0 ..< 4 * self.layerCount).map { (_) in
		return (0 ..< self.hiddenSize * self.hiddenSize).map { (_) in
			return Float.random(in: -1 ... 1)
		}
	}
	
	private lazy var biasesDataArrays = (0 ..< 4 * self.layerCount).map { (_) in
		return (0 ..< self.hiddenSize).map { (_) in
			return 0 as Float
		}
	}
	
	/// Creates an unconfigured LSTM layer.
	/// - Parameters:
	///   - hiddenSize: The dimension of the hidden state. This is also the primary axis of the shape of the output tensor.
	///   - layerCount: The number of internal layers.
	///   - returnsSequences: Whether the final internal layer should return the entire sequence of intermediate results instead of just the final result.
	public init(hiddenSize: Int, layerCount: Int = 1, returnsSequences: Bool = false) {
		precondition(hiddenSize > 0, "Invalid hidden size")
		precondition(layerCount > 0, "Invalid layer count")
		self.hiddenSize = hiddenSize
		self.layerCount = layerCount
		self.returnsSequences = returnsSequences
	}
	
	public func storeWeights(in weightsContainer: inout WeightsContainer) throws {
		try weightsContainer.store(self.inputWeights, self.hiddenWeights, self.biases)
	}
	
	public mutating func loadWeights(from weightsContainer: WeightsContainer, at index: Int, on internalDevice: MLCDevice) throws {
		let internalTensors = weightsContainer[index]
		let inputWeightsBoundary = 4 * self.layerCount
		let hiddenWeightsBoundary = inputWeightsBoundary + 4 * self.layerCount
		let biasesBoundary = hiddenWeightsBoundary + 4 * self.layerCount
		self.inputWeights = try internalTensors[0 ..< inputWeightsBoundary].internalTensors(on: internalDevice)
		self.hiddenWeights = try internalTensors[inputWeightsBoundary ..< hiddenWeightsBoundary].internalTensors(on: internalDevice)
		self.biases = try internalTensors[hiddenWeightsBoundary ..< biasesBoundary].internalTensors(on: internalDevice)
	}
	
	public mutating func configure(inputShape: Tensor.Shape, on internalDevice: MLCDevice) {
		self.inputShape = inputShape
		self.outputShape = Tensor.Shape(primaryAxis: self.hiddenSize, secondaryAxis: self.inputShape.secondaryAxis, batchSize: self.inputShape.batchSize)
		if self.inputWeights == nil {
			self.inputWeights = (0 ..< 4 * self.layerCount).map { (index) -> MLCTensor in
				MLCTensor(dataArray: self.inputWeightsDataArrays[index], shape: [self.inputShape.primaryAxis, self.hiddenSize], on: internalDevice)
			}
		}
		if self.hiddenWeights == nil {
			self.hiddenWeights = (0 ..< 4 * self.layerCount).map { (index) -> MLCTensor in
				return MLCTensor(dataArray: self.hiddenWeightsDataArrays[index], shape: [self.hiddenSize, self.hiddenSize], on: internalDevice)
			}
		}
		if self.biases == nil {
			self.biases = (0 ..< 4 * self.layerCount).map { (index) -> MLCTensor in
				return MLCTensor(dataArray: self.biasesDataArrays[index], shape: [self.hiddenSize], on: internalDevice)
			}
		}
		let descriptor = MLCLSTMDescriptor(inputSize: self.inputShape.primaryAxis, hiddenSize: self.hiddenSize, layerCount: self.layerCount, usesBiases: true, batchFirst: true, isBidirectional: false, returnsSequences: self.returnsSequences, dropout: 0)
		guard let internalLayer = MLCLSTMLayer(descriptor: descriptor, inputWeights: self.inputWeights, hiddenWeights: self.hiddenWeights, biases: self.biases) else {
			fatalError("Construction of LSTM layer failed")
		}
		self.internalLayer = internalLayer
	}
	
	public func checkTrainingCompatibility(on device: TrainingComputeDevice) -> Bool {
		return device != .gpu /// `MLCLSTMLayer` currently has issues with retrieving trained weights from a GPU due to a bug in ML Compute.
	}
	
}
