//
//  TrainingData.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 12/9/21.
//

/// A structure that holds input and target tensors for training.
public struct TrainingData {
	
	/// The input tensors.
	public private(set) var inputTensors: [Tensor]
	
	/// The target tensors.
	public private(set) var targetTensors: [Tensor]
	
	/// Creates a `TrainingData` structure.
	/// - Parameters:
	///   - inputTensors: The input tensors.
	///   - targetTensors: The target tensors.
	public init(inputTensors: [Tensor], targetTensors: [Tensor]) {
		precondition(inputTensors.count > 0, "Must supply at least one input tensor")
		precondition(inputTensors.count == targetTensors.count, "Must supply the same number of input tensors and target tensors")
		let inputSameShape = inputTensors
			.dropFirst()
			.allSatisfy { (tensor) in
				return tensor.shape == inputTensors.first!.shape
			}
		precondition(inputSameShape, "All input tensors must have the same shape")
		let targetSameShape = targetTensors
			.dropFirst()
			.allSatisfy { (tensor) in
				return tensor.shape == targetTensors.first!.shape
			}
		precondition(targetSameShape, "All target tensors must have the same shape")
		precondition(inputTensors.first!.shape.batchSize == targetTensors.first!.shape.batchSize, "Input tensors and target tensors must have the same batch size")
		self.inputTensors = inputTensors
		self.targetTensors = targetTensors
	}
	
	/// Automatically combines the input and target tensors into batches for batched training.
	/// - Parameter batchSize: The size of each batch.
	public mutating func autobatch(size batchSize: Int) {
		self.inputTensors.autobatch(size: batchSize)
		self.targetTensors.autobatch(size: batchSize)
	}
	
}
