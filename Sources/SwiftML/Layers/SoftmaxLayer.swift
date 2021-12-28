//
//  SoftmaxLayer.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 3/11/21.
//

import MLCompute

/// A softmax layer.
public struct SoftmaxLayer: Layer {
	
	public private(set) var internalLayer: MLCLayer!
	
	public private(set) var inputShape: Tensor.Shape!
	
	public private(set) var outputShape: Tensor.Shape!
	
	private let doUseLogVariant: Bool
	
	/// Creates an unconfigured softmax layer.
	/// - Parameter doUseLogVariant: Whether to use the logarithmic variant of the softmax operation.
	public init(useLogVariant doUseLogVariant: Bool = false) {
		self.doUseLogVariant = doUseLogVariant
	}
	
	public mutating func configure(inputShape: Tensor.Shape, on _: MLCDevice) {
		self.inputShape = inputShape
		self.outputShape = self.inputShape
		self.internalLayer = MLCSoftmaxLayer(operation: self.doUseLogVariant ? .logSoftmax : .softmax)
	}
	
}
