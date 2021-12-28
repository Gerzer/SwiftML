//
//  ActivationLayer.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 12/28/21.
//

import MLCompute

/// An activation layer.
public struct ActivationLayer: Layer {
	
	public enum ActivationType {
		
		case celu(alpha: Float? = nil)
		
		case clamp(min: Float, max: Float)
		
		case elu(Float? = nil)
		
		case gelu
		
		case hardShrink(lambda: Float? = nil)
		
		case hardSigmoid
		
		case hardSwish
		
		case leakyReLU(negativeSlope: Float? = nil)
		
		case linear(scale: Float, bias: Float)
		
		case logSigmoid
		
		case relu
		
		case relun(alpha: Float, beta: Float)
		
		case relu6
		
		case selu
		
		case sigmoid
		
		case softPlus(beta: Float? = nil)
		
		case softShrink(lambda: Float? = nil)
		
		case softSign
		
		case tanh
		
		case tanhShrink
		
		case threshold(Float, replacement: Float)
		
		fileprivate func select() -> MLCActivationLayer {
			switch self {
			case .celu(alpha: nil):
				return .celu
			case .celu(alpha: let alpha?):
				return .celu(a: alpha)
			case .clamp(min: let min, max: let max):
				return .clamp(min: min, max: max)
			case .elu(alpha: nil):
				return .elu
			case .elu(alpha: let alpha?):
				return .elu(a: alpha)
			case .gelu:
				return .gelu
			case .hardShrink(lambda: nil):
				return .hardShrink
			case .hardShrink(lambda: let lambda?):
				return .hardShrink(a: lambda)
			case .hardSigmoid:
				return .hardSigmoid
			case .hardSwish:
				return .hardSwish
			case .leakyReLU(negativeSlope: nil):
				return .leakyReLU
			case .leakyReLU(negativeSlope: let negativeSlope?):
				return .leakyReLU(negativeSlope: negativeSlope)
			case .linear(scale: let scale, bias: let bias):
				return .linear(scale: scale, bias: bias)
			case .logSigmoid:
				return .logSigmoid
			case .relu:
				return .relu
			case .relun(alpha: let alpha, beta: let beta):
				return .relun(a: alpha, b: beta)
			case .relu6:
				return .relu6
			case .selu:
				return .selu
			case .sigmoid:
				return .sigmoid
			case .softPlus(beta: nil):
				return .softPlus
			case .softPlus(beta: let beta?):
				return .softPlus(beta: beta)
			case .softShrink(lambda: nil):
				return .softShrink
			case .softShrink(lambda: let lambda?):
				return .softShrink(a: lambda)
			case .softSign:
				return .softSign
			case .tanh:
				return .tanh
			case .tanhShrink:
				return .tanhShrink
			case .threshold(let value, replacement: let replacement):
				return .threshold(value, replacement: replacement)
			}
		}
		
	}
	
	public private(set) var internalLayer: MLCLayer!
	
	public private(set) var inputShape: Tensor.Shape!
	
	public private(set) var outputShape: Tensor.Shape!
	
	private let activationType: ActivationType
	
	/// Creates an unconfigured activation layer.
	/// - Parameter activationType: The type of activation to use, including any relevant parameters.
	public init(_ activationType: ActivationType) {
		self.activationType = activationType
	}
	
	public mutating func configure(inputShape: Tensor.Shape, on _: MLCDevice) {
		self.inputShape = inputShape
		self.outputShape = self.inputShape
		self.internalLayer = self.activationType.select()
	}
	
}
