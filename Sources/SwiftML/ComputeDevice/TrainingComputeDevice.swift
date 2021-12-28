//
//  TrainingComputeDevice.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 12/11/21.
//

import MLCompute

/// A compute device on which to execute a ``Graph`` instance for training.
public enum TrainingComputeDevice: ComputeDevice {
	
	/// A CPU compute device.
	/// - Note: This device should always be available.
	case cpu
	
	/// A GPU compute device.
	/// - Warning: This device might not always be available.
	case gpu
	
	public func select() throws -> MLCDevice {
		switch self {
		case .cpu:
			return .cpu()
		case .gpu:
			guard let internalDevice = MLCDevice.gpu() else {
				throw DeviceError.noGPUDetected
			}
			return internalDevice
		}
	}
	
}
