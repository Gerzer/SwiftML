//
//  Tensor.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 3/11/21.
//

import MLCompute

/// A three-dimensional tensor of `Float` data.
public struct Tensor: CustomStringConvertible {
	
	/// The shape of a three-dimensional tensor.
	public struct Shape: Equatable {
		
		/// The length of the primary axis.
		public internal(set) var primaryAxis: Int
		
		/// The length of the secondary axis.
		public internal(set) var secondaryAxis: Int
		
		/// The batch size.
		public internal(set) var batchSize: Int
		
		/// The total number of elements in the tensor.
		public var flatLength: Int {
			get {
				let primaryAxis = self.primaryAxis > 0 ? self.primaryAxis : 1
				let secondaryAxis = self.secondaryAxis > 0 ? self.secondaryAxis : 1
				let batchSize = self.batchSize > 0 ? self.batchSize : 1
				return primaryAxis * secondaryAxis * batchSize
			}
		}
		
		var shapeArray: [Int] {
			get {
				if self.batchSize == 0 {
					if self.secondaryAxis == 0 {
						return [self.primaryAxis]
					} else {
						return [self.secondaryAxis, self.primaryAxis]
					}
				} else {
					return [self.batchSize, self.secondaryAxis, self.primaryAxis]
				}
			}
		}
		
		/// Creates a representation of the shape of a three-dimensional tensor.
		/// - Parameters:
		///   - primaryAxis: The length of the primary axis.
		///   - secondaryAxis: The length of the secondary axis.
		///   - batchSize: The batch size.
		public init(primaryAxis: Int, secondaryAxis: Int, batchSize: Int = 1) {
			precondition(primaryAxis > 0, "Invalid primary axis")
			precondition(secondaryAxis > 0, "Invalid secondary axis")
			precondition(batchSize > 0, "Invalid batch size")
			self.primaryAxis = primaryAxis
			self.secondaryAxis = secondaryAxis
			self.batchSize = batchSize
		}
		
		init(shapeArray: [Int]) {
			assert(shapeArray.count == 3, "Invalid shape array")
			self.init(primaryAxis: shapeArray[2], secondaryAxis: shapeArray[1], batchSize: shapeArray[0])
		}
		
		fileprivate init(unsafelyWithShapeArray shapeArray: [Int]) {
			switch shapeArray.count {
			case 1:
				self.primaryAxis = shapeArray[0]
				self.secondaryAxis = 0
				self.batchSize = 0
			case 2:
				self.primaryAxis = shapeArray[1]
				self.secondaryAxis = shapeArray[0]
				self.batchSize = 0
			case 3:
				self.primaryAxis = shapeArray[2]
				self.secondaryAxis = shapeArray[1]
				self.batchSize = shapeArray[0]
			default:
				fatalError("Unexpected length of shape array")
			}
		}
		
	}
	
	let shape: Shape
	
	private var data: [[[Float]]] {
		get {
			var value: [[[Float]]] = [[[0] * self.shape.primaryAxis] * self.shape.secondaryAxis] * self.shape.batchSize
			for batchIndex in 0 ..< self.shape.batchSize {
				let batchOffset = self.shape.primaryAxis * self.shape.secondaryAxis * batchIndex
				for secondaryIndex in 0 ..< self.shape.secondaryAxis {
					let secondaryOffset = self.shape.primaryAxis * secondaryIndex
					for primaryIndex in 0 ..< self.shape.primaryAxis {
						let offset = batchOffset + secondaryOffset + primaryIndex
						value[batchIndex][secondaryIndex][primaryIndex] = self.flatData[offset]
					}
				}
			}
			return value
		}
		set {
			self.flatData = newValue.flatMap { (outerArray) in
				return outerArray.flatMap { (innerArray) in
					return innerArray
				}
			}
		}
	}
	
	fileprivate private(set) var flatData: [Float]
	
	public var batchElements: [Tensor] {
		get {
			return self.data.map { (outerArray) in
				let shape = Shape(primaryAxis: self.shape.primaryAxis, secondaryAxis: self.shape.secondaryAxis)
				return Tensor(data: [outerArray], shape: shape)
			}
		}
	}
	
	/// A textual, human-readable representation of the data.
	public var description: String {
		get {
			var description = ""
			let formatter = NumberFormatter()
			self.data
				.enumerated()
				.forEach { (offset, secondaryData) in
					description.append("Batch element \(offset):\n")
					for primaryData in secondaryData {
						for element in primaryData {
							if !element.isNaN {
								let fractionDigits = 9 - element.intDigitsCount
								let actualFractionDigits = element < 0 ? fractionDigits - 1 : fractionDigits
								formatter.minimumIntegerDigits = element.intDigitsCount
								formatter.maximumIntegerDigits = element.intDigitsCount
								formatter.minimumFractionDigits = actualFractionDigits
								formatter.maximumFractionDigits = actualFractionDigits
							}
							description.append("\(formatter.string(from: NSNumber(value: element))!)\t\t")
						}
						description.removeLast()
						description.append("\n")
					}
					description.append("\n")
				}
			description.removeLast()
			return description
		}
	}
	
	/// Creates a tensor.
	///
	/// All data elements are initialized to `0`.
	/// - Parameter shape: The shape of the tensor.
	public init(shape: Shape) {
		self.shape = shape
		self.flatData = [0] * self.shape.flatLength
	}
	
	/// Creates a tensor.
	///
	/// All data elements are initialized to `0`.
	/// - Parameters:
	///   - primaryAxis: The length of the primary axis.
	///   - secondaryAxis: The length of the secondary axis.
	///   - batchSize: The batch size.
	public init(primaryAxis: Int, secondaryAxis: Int, batchSize: Int = 1) {
		let shape = Shape(primaryAxis: primaryAxis, secondaryAxis: secondaryAxis, batchSize: batchSize)
		self.init(shape: shape)
	}
	
	/// Creates a tensor.
	/// - Parameters:
	///   - flatData: A flat array representation of the three-dimensional data.
	///   - shape: The shape of the tensor.
	public init(flatData: [Float], shape: Shape) {
		precondition(flatData.count == shape.flatLength, "Shape mismatch")
		self.shape = shape
		self.flatData = flatData
	}
	
	init(data: [[[Float]]], shape: Shape) {
		// TODO: Add precondition checks
		self.init(shape: shape)
		self.data = data
	}
	
	init(repeating oldTensor: Tensor, batchSize newBatchSize: Int) {
		precondition(oldTensor.shape.batchSize == 1, "Invalid tensor shape")
		precondition(newBatchSize > 0, "Invalid batch size")
		let newFlatData = oldTensor.flatData * newBatchSize
		let newShape = Tensor.Shape(primaryAxis: oldTensor.shape.primaryAxis, secondaryAxis: oldTensor.shape.secondaryAxis, batchSize: newBatchSize)
		self.init(flatData: newFlatData, shape: newShape)
	}
	
	init(from internalTensor: MLCTensor) throws {
		let count = internalTensor.descriptor.shape.reduce(1, *)
		var flatData = [Float](repeating: 0, count: count)
		let success = flatData.withUnsafeMutableBufferPointer { (pointer) in
			return internalTensor.copyDataFromDeviceMemory(toBytes: pointer.baseAddress!, length: count * MemoryLayout<Float>.size, synchronizeWithDevice: true)
		}
		assert(success)
		let shape = Shape(unsafelyWithShapeArray: internalTensor.descriptor.shape)
		self.init(flatData: flatData, shape: shape)
	}
	
	public subscript(primaryIndex: Int, secondaryIndex: Int, batchIndex: Int) -> Float {
		get {
			return self.data[batchIndex][secondaryIndex][primaryIndex]
		} set {
			self.data[batchIndex][secondaryIndex][primaryIndex] = newValue
		}
	}
	
	func internalTensor(on internalDevice: MLCDevice) throws -> MLCTensor {
		return self.flatData.withUnsafeBytes { (pointer) in
			let internalTensorData = MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!, length: self.shape.flatLength * MemoryLayout<Float>.size)
			let internalTensor = MLCTensor(shape: self.shape.shapeArray)
			internalTensor.bindAndWriteData(internalTensorData, to: internalDevice)
			return internalTensor
		}
	}
	
	func internalTensorData() -> MLCTensorData {
		return self.flatData.withUnsafeBytes { (pointer) -> MLCTensorData in
			return MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!, length: self.shape.flatLength * MemoryLayout<Float>.size)
		}
	}
	
}

extension Array where Element == Tensor {
	
	mutating func autobatch(size batchSize: Int) {
		precondition(batchSize > 0, "Invalid batch size")
		precondition(self.count > 0, "Must supply at least one tensor")
		precondition(self.count.isMultiple(of: batchSize), "Invalid batch size for number of supplied tensors")
		let unbatched = self.allSatisfy { (tensor) in
			return tensor.shape.batchSize == 1
		}
		precondition(unbatched, "Must supply unbatched tensors")
		let shape = Tensor.Shape(primaryAxis: self.first!.shape.primaryAxis, secondaryAxis: self.first!.shape.secondaryAxis, batchSize: batchSize)
		let tensors = (0 ..< self.count / batchSize).map { (baseIndex) -> Tensor in
			var flatData = [Float]()
			for offset in 0 ..< batchSize {
				flatData.append(contentsOf: self[baseIndex + offset].flatData)
			}
			return Tensor(flatData: flatData, shape: shape)
		}
		self = tensors
	}
	
}

extension Collection where Element == Tensor {
	
	func internalTensors(on internalDevice: MLCDevice) throws -> [MLCTensor] {
		return try self.map { (tensor) in
			return try tensor.internalTensor(on: internalDevice)
		}
	}
	
}
