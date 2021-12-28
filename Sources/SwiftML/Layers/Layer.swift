//
//  Layer.swift
//  SwiftML
//
//  Created by Gabriel Jacoby-Cooper on 7/11/20.
//

import MLCompute

/// An executable layer, implemented as a wrapper around an internal ML Compute `MLCLayer` instance.
public protocol Layer {
	
	/// The internal ML Compute layer object.
	///
	/// This should be an instance of a concrete subclass of `MLCLayer`, not an instance of `MLCLayer` itself.
	/// - Important: Conforming types should restrict write access for this property to the `private` scope.
	var internalLayer: MLCLayer! { get }
	
	/// The shape of the input tensor.
	/// 
	/// - Important: Conforming types should restrict write access for this property to the `private` scope.
	var inputShape: Tensor.Shape! { get }
	
	/// The shape of the output tensor.
	///
	/// - Important: Conforming types should restrict write access for this property to the `private` scope.
	var outputShape: Tensor.Shape! { get }
	
	/// Stores the layer’s weights in a ``WeightsContainer`` instance.
	///
	/// A default implementation that does nothing is provided; this is useful for layers that don’t have any weights. All layers that *do* have weights should manually implement this method.
	/// - Parameter weightsContainer: The weights container.
	/// - Warning: Don’t call this method directly.
	func storeWeights(in weightsContainer: inout WeightsContainer) throws
	
	/// Loads the layer’s weights from a ``WeightsContainer`` instance.
	///
	/// A default implementation that does nothing is provided; this is useful for layers that don’t have any weights. All layers that *do* have weights should manually implement this method.
	/// - Parameters:
	///   - weightsContainer: The weights container.
	///   - index: The index at which the weights are stored in the weights container.
	///   - internalDevice: The device onto which to load the weights.
	/// - Warning: Don’t call this method directly.
	mutating func loadWeights(from weightsContainer: WeightsContainer, at index: Int, on internalDevice: MLCDevice) throws
	
	/// Sets the ``internalLayer``, ``inputShape``, and ``outputShape`` properties and configures the internal ML Compute layer object for future execution.
	///
	/// This method shouldn’t actually execute the layer; execution is handled by the ``Graph`` class. SwiftML guarantees that this method will always be invoked before the layer is actually executed by the framework.
	/// - Parameters:
	///   - inputShape: The expected shape of the input tensor.
	///   - internalDevice: The device on which this layer will be executed.
	/// - Warning: Don’t call this method directly.
	mutating func configure(inputShape: Tensor.Shape, on internalDevice: MLCDevice)
	
	/// Checks whether this layer can be trained on the specified device.
	///
	/// A layer might be incompatible with a particular device for a variety of reasons, including the presence of known issues in the underlying ML Compute framework. Hence, the compatibility result might change in the future as issues are fixed and APIs are improved.
	///
	/// A default implementation that always returns `true` is provided; this is useful for layers that don’t have any known compatibility issues. All layers that *do* have known compatibility issues should manually implement this method.
	/// - Parameter device: The device for which to check compatibility.
	/// - Returns: Whether this layer can be trained on the specified device.
	func checkTrainingCompatibility(on device: TrainingComputeDevice) -> Bool
	
	/// Checks whether inference can be performed with this layer on the specified device.
	///
	/// A layer might be incompatible on a particular device for a variety of reasons, including the presence of known issues in the underlying ML Compute framework. Hence, the compatibility result might change in the future as issues are fixed and APIs are improved.
	///
	/// A default implementation that always returns `true` is provided; this is useful for layers that don’t have any known compatibility issues. All layers that *do* have known compatibility issues should manually implement this method.
	/// - Parameter device: The device for which to check compatibility.
	/// - Returns: Whether inference can be performed with this layer on the specified device.
	func checkInferenceCompatibility(on device: InferenceComputeDevice) -> Bool
	
}

public extension Layer {
	
	func storeWeights(in _: inout WeightsContainer) throws { }
	
	func loadWeights(from _: WeightsContainer, at _: Int, on _: MLCDevice) throws { }
	
	func checkTrainingCompatibility(on device: TrainingComputeDevice) -> Bool {
		return true
	}
	
	func checkInferenceCompatibility(on device: InferenceComputeDevice) -> Bool {
		return true
	}
	
}
