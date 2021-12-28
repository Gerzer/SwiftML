//
//  FullyConnectedLayer.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 3/11/21.
//

import MLCompute

/// A fully-connected layer.
///
/// This type of layer is sometimes known as a “dense” layer.
public struct FullyConnectedLayer: Layer {
	
	public private(set) var internalLayer: MLCLayer!
	
	public private(set) var inputShape: Tensor.Shape!
	
	public private(set) var outputShape: Tensor.Shape!
	
	private let outputSize: Int
	
	private var weights: MLCTensor! = nil
	
	private var biases: MLCTensor! = nil
	
	private lazy var weightsDataArray = (0 ..< self.inputShape.primaryAxis * self.outputSize).map { (_) in
		return Float.random(in: -1 ... 1)
	}
	
	private lazy var biasesDataArray = (0 ..< self.outputSize).map { (_) in
		return 0 as Float
	}
	
	/// Creates an unconfigured fully-connected layer
	/// - Parameter outputSize: The dimension of the output. This is also the primary axis of the shape of the output tensor.
	public init(outputSize: Int) {
		precondition(outputSize > 0, "Invalid output size")
		self.outputSize = outputSize
	}
	
	public func storeWeights(in weightsContainer: inout WeightsContainer) throws {
		try weightsContainer.store(self.weights, self.biases)
	}
	
	public mutating func loadWeights(from weightsContainer: WeightsContainer, at index: Int, on internalDevice: MLCDevice) throws {
		let internalTensors = weightsContainer[index]
		self.weights = try internalTensors[0].internalTensor(on: internalDevice)
		self.biases = try internalTensors[1].internalTensor(on: internalDevice)
	}
	
	public mutating func configure(inputShape: Tensor.Shape, on internalDevice: MLCDevice) {
		self.inputShape = inputShape
		self.outputShape = Tensor.Shape(primaryAxis: self.outputSize, secondaryAxis: inputShape.secondaryAxis, batchSize: inputShape.batchSize)
		if self.weights == nil {
//			self.weights = MLCTensor(shape: [1, self.inputShape.primaryAxis * self.outputSize], randomInitializerType: .glorotUniform)
			self.weights = MLCTensor(dataArray: self.weightsDataArray, shape: [1, self.inputShape.primaryAxis * self.outputSize], on: internalDevice)
		}
		if self.biases == nil {
//			self.biases = MLCTensor(shape: [1, self.outputSize], fillWithData: 0, dataType: .float32)
			self.biases = MLCTensor(dataArray: self.biasesDataArray, shape: [1, self.outputSize], on: internalDevice)
		}
		let descriptor = MLCConvolutionDescriptor(kernelSizes: (height: 1, width: 1), inputFeatureChannelCount: self.inputShape.primaryAxis, outputFeatureChannelCount: self.outputSize)
		guard let internalLayer = MLCFullyConnectedLayer(weights: self.weights, biases: self.biases, descriptor: descriptor) else {
			fatalError("Construction of fully-connected layer failed")
		}
		self.internalLayer = internalLayer
	}
	
}
