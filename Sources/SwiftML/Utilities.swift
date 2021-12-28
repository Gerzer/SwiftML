//
//  Utilities.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 3/8/21.
//

import MLCompute
import Foundation

/// An error that’s thrown when a particular device that was requested is unavailable.
public enum DeviceError: Error {
	
	case noGPUDetected, noANEDetected
	
}

public enum TrainingError: Error {
	
	case incompatible, alreadyTrained
	
}

public enum InferenceError: Error {
	
	case incompatible, untrained
	
}

extension MLCTensor {
	
	enum DataError: Error {
		
		case uninitialized
		
	}
	
	convenience init(dataArray: [Float], shape: [Int], on device: MLCDevice) {
		precondition(dataArray.count == shape.reduce(1, *))
		self.init(shape: shape)
		let data = dataArray.withUnsafeBytes { (pointer) in
			return MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!, length: dataArray.count * MemoryLayout<Float>.size)
		}
		self.bindAndWriteData(data, to: device)
	}
	
	func dataArray<Scalar>(as _: Scalar.Type) throws -> [Scalar] where Scalar: Numeric {
		let count = self.descriptor.shape.reduce(into: 1) { (result, value) in
			result *= value
		}
		var array = [Scalar](repeating: 0, count: count)
		self.synchronizeData()
		_ = try array.withUnsafeMutableBytes { (pointer) in
			guard let data = self.data else {
				throw DataError.uninitialized
			}
			data.copyBytes(to: pointer)
		}
		return array
	}
	
}

extension Array {
	
	static func * (_ array: Self, _ count: Int) -> Self {
		let newArray = (0 ..< count).flatMap { (_) in
			return array
		}
		return Self(newArray)
	}
	
}

extension Float {
	
	var intDigitsCount: Int {
		get {
			return max(Int(log10(max(abs(self), 1))), 0) + 1
		}
	}
	
}
