//
//  InferenceComputeDevice.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 3/11/21.
//

import MLCompute

/// A compute device on which to execute a ``Graph`` instance for inference.
public enum InferenceComputeDevice: ComputeDevice {
	
	/// A CPU compute device.
	/// - Note: This device should always be available.
	case cpu
	
	/// A GPU compute device.
	/// - Warning: This device might not always be available.
	case gpu
	
	/// An Apple Neural Engine (ANE) compute device.
	///
	/// Some layers, such as ``LSTMLayer``, are unsupported on the ANE and might instead be executed on either the CPU or the GPU. You might see warnings from the underlying ML Compute framework in the console output if this happens.
	/// - Warning: This device might not always be available.
	case ane
	
	public func select() throws -> MLCDevice {
		switch self {
		case .cpu:
			return .cpu()
		case .gpu:
			guard let internalDevice = MLCDevice.gpu() else {
				throw DeviceError.noGPUDetected
			}
			return internalDevice
		case .ane:
			if #available(macOS 12.0, *) {
				guard let internalDevice = MLCDevice.ane() else {
					throw DeviceError.noANEDetected
				}
				return internalDevice
			} else {
				throw DeviceError.noANEDetected
			}
		}
	}
	
}
