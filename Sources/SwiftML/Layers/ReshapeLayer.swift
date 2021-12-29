//
//  ReshapeLayer.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 12/29/21.
//

import MLCompute

/// A reshape layer.
public struct ReshapeLayer: Layer {
	
	public private(set) var internalLayer: MLCLayer!
	
	public private(set) var inputShape: Tensor.Shape!
	
	public private(set) var outputShape: Tensor.Shape!
	
	private var newShape: Tensor.Shape
	
	private let didSetDefaultBatchSize: Bool
	
	/// Creates a reshape layer.
	/// - Parameter newShape: The new shape of the output.
	/// - Important: The number of elements that’s specified by the new shape must be the same as the number of elements in the input tensor during training. Furthermore, the new batch size must be the same as the batch size of the input shape during training.
	public init(newShape: Tensor.Shape) {
		self.newShape = newShape
		self.didSetDefaultBatchSize = false
	}
	
	/// Creates a reshape layer.
	/// - Parameters:
	///   - newPrimaryAxis: The new primary axis of the output shape.
	///   - newSecondaryAxis: The new secondary axis of the output shape.
	/// - Important: The number of elements that’s specified by the new shape must be the same as the number of elements in the input tensor during training.
	public init(newPrimaryAxis: Int, newSecondaryAxis: Int) {
		self.newShape = Tensor.Shape(primaryAxis: newPrimaryAxis, secondaryAxis: newSecondaryAxis)
		self.didSetDefaultBatchSize = true
	}
	
	public mutating func configure(inputShape: Tensor.Shape, on _: MLCDevice) {
		if self.didSetDefaultBatchSize {
			self.newShape.batchSize = inputShape.batchSize
		}
		precondition(inputShape.batchSize == newShape.batchSize, "Input and output batch sizes must be the same")
		precondition(inputShape.shapeArray.reduce(1, *) == newShape.shapeArray.reduce(1, *), "Input and output tensors must have the same number of elements")
		self.inputShape = inputShape
		self.outputShape = self.newShape
		guard let internalLayer = MLCReshapeLayer(shape: self.outputShape.shapeArray) else {
			fatalError("Construction of reshape layer failed")
		}
		self.internalLayer = internalLayer
	}
	
}
