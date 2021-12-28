//
//  InternalTensors.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 12/11/21.
//

import MLCompute

/// A protocol to which various representations of internal ML Compute tensors conform.
/// - Warning: Don’t use this protocol directly.
public protocol InternalTensors {
	
	/// Gets the tensors as an array of ``Tensor`` instances.
	/// - Returns: The tensors.
	/// - Warning: Don’t call this method directly.
	func getTensors() throws -> [Tensor]
	
	func synchronizeData() -> Bool
	
}

extension MLCTensor: InternalTensors {
	
	public func getTensors() throws -> [Tensor] {
		return [try Tensor(from: self)]
	}
	
}

extension Array: InternalTensors where Element: InternalTensors {
	
	public func getTensors() throws -> [Tensor] {
		return try self.flatMap { (internalTensors) in
			return try internalTensors.getTensors()
		}
	}
	
	public func synchronizeData() -> Bool {
		return self.allSatisfy { (internalTensors) in
			return internalTensors.synchronizeData()
		}
	}
	
}
