//
//  ComputeDevice.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 12/12/21.
//

import MLCompute

/// A protocol to which enumerations that enable the selection of a particular compute device should conform.
///
/// The built-in concrete types that conform to this protocol are ``InferenceComputeDevice`` and ``TrainingComputeDevice``.
public protocol ComputeDevice {
	
	/// Selects a specific internal ML Compute device.
	///
	/// Device selection is handled internally by SwiftML.
	/// - Returns: The internal ML Compute device.
	/// - Warning: Don’t call this method directly.
	func select() throws -> MLCDevice
	
}
