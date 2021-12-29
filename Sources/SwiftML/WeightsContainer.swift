//
//  WeightsContainer.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 12/11/21.
//

/// A structure that contains weight tensors.
public struct WeightsContainer {
	
	private var tensors = [[Tensor]]()
	
	/// Gets the array of weight tensors at the specified index.
	/// - Parameter index: The index of the array of tensors to retrieve. This should typically be the index of the layer in the graph.
	/// - Returns: The array of tensors.
	public subscript(_ index: Int) -> [Tensor] {
		get {
			return self.tensors[index]
		}
	}
	
	/// Stores the specified internal tensors in this container.
	/// - Parameter internalTensors: The internal tensors to store.
	public mutating func store(_ internalTensors: InternalTensors...) throws {
		let newTensors = try internalTensors.flatMap { (element) in
			return try element.getTensors()
		}
		self.tensors.append(newTensors)
	}
	
}
