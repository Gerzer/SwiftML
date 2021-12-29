//
//  ReshapeLayer.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 12/29/21.
//

import MLCompute

public struct ReshapeLayer: Layer {
	
	public private(set) var internalLayer: MLCLayer!
	
	public private(set) var inputShape: Tensor.Shape!
	
	public private(set) var outputShape: Tensor.Shape!
	
	private var newShape: Tensor.Shape
	
	private let didSetDefaultBatchSize: Bool
	
	public init(newShape: Tensor.Shape) {
		self.newShape = newShape
		self.didSetDefaultBatchSize = false
	}
	
	public init(newPrimaryAxis primaryAxis: Int, newSecondaryAxis secondaryAxis: Int) {
		self.newShape = Tensor.Shape(primaryAxis: primaryAxis, secondaryAxis: secondaryAxis)
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
